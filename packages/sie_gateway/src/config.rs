use std::collections::HashMap;
use std::env;

use serde::Deserialize;

use crate::state::demand_tracker::PhysicalLaneCatalog;
use crate::types::pool::PoolSpec;

/// Modal platform proxy-auth credential (superlinked/sie-internal #1740).
///
/// When a managed ingress endpoint (config service, generation-lane `/gen`
/// WebSocket, OTLP collector) is deployed with `requires_proxy_auth=True`,
/// Modal rejects any request missing a valid `Modal-Key` / `Modal-Secret`
/// pair AT THE PLATFORM EDGE, before the request reaches app code. The
/// gateway's config client presents this pair ALONGSIDE the in-app admin
/// bearer (defense in depth — neither replaces the other).
///
/// Read from `SIE_MODAL_PROXY_TOKEN_ID` / `SIE_MODAL_PROXY_TOKEN_SECRET`.
/// `Some` only when BOTH are non-empty; a half-set pair is a misconfiguration,
/// not partial auth, so it is treated as absent. Absent means the headers are
/// never sent, so self-host / dev (no Modal edge) is untouched.
///
/// `Debug` is hand-written to REDACT both halves: `Config` derives `Debug` and
/// is `{:?}`-formatted at startup and in tests, so a derived impl here would
/// leak the credential into logs (the `Option` already conveys presence).
#[derive(Clone)]
pub struct ModalProxyToken {
    pub key: String,
    pub secret: String,
}

impl std::fmt::Debug for ModalProxyToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModalProxyToken")
            .field("key", &"<redacted>")
            .field("secret", &"<redacted>")
            .finish()
    }
}

// `Debug` is hand-written (below) rather than derived: several fields hold
// secret material (`auth_tokens`, `admin_token`, `config_service_token`, and
// `config_modal_proxy_token`) and `Config` is `{:?}`-formatted at startup, so a
// derived impl would leak credentials into logs. clippy's
// `missing_fields_in_debug` keeps the manual impl exhaustive as fields are added.
#[derive(Clone)]
pub struct Config {
    // Server
    pub host: String,
    pub port: u16,

    // Discovery
    pub worker_urls: Vec<String>,
    pub use_kubernetes: bool,
    pub k8s_namespace: String,
    pub k8s_service: String,
    pub k8s_port: u16,

    // Features
    pub health_mode: String,

    // NATS
    pub nats_url: String,
    /// Trusted-producer allowlist for `sie.config.models._all`. Defaults
    /// to `["sie-config"]`. Incoming `ConfigNotification`s whose
    /// `producer_id` is not in this list are dropped (neither the epoch
    /// counter nor the registry is touched). Override via
    /// `SIE_NATS_CONFIG_TRUSTED_PRODUCERS` (CSV), or disable validation
    /// entirely with `SIE_NATS_CONFIG_TRUST_ANY_PRODUCER=true` (intended
    /// for local dev / single-node test clusters only).
    pub nats_config_trusted_producers: Vec<String>,

    // Auth
    pub auth_mode: String,
    pub auth_tokens: Vec<String>,
    pub admin_token: String,
    /// Opt-in bypass for operational surfaces (`/`, `/health`, `/ws/*`)
    /// when auth is enabled. Kubernetes probes
    /// (`/healthz`, `/readyz`) are always exempt regardless. Defaults
    /// to `false` (fail-closed); set `SIE_AUTH_EXEMPT_OPERATIONAL=true`
    /// only when these endpoints are already network-isolated (e.g.
    /// internal ClusterIP with no ingress).
    pub auth_exempt_operational: bool,

    // Logging
    pub log_level: String,
    pub json_logs: bool,

    // Feature toggles
    pub enable_pools: bool,
    pub hot_reload: bool,
    pub watch_polling: bool,
    pub multi_router: bool,

    // Tuning
    pub request_timeout: f64,
    pub max_stream_pending: u64,
    pub stream_max_age_s: u64,

    // Configured GPUs (survives scale-to-zero)
    pub configured_gpus: Vec<String>,
    // Pre-computed lowercase→original map for GPU profile resolution (avoids HashMap rebuild per request)
    pub gpu_profile_map: HashMap<String, String>,
    /// Exact Helm/KEDA physical lane catalog. Scale-driving state and
    /// telemetry labels fail closed unless their full tuple resolves here.
    pub configured_physical_lanes: PhysicalLaneCatalog,

    // Helm-declared queue pools that are long-lived admission boundaries.
    // API-created pools remain lease-based; these specs do not expire.
    pub static_queue_pools: Vec<PoolSpec>,

    // Job/friendly model aliases: lowercase alias → canonical model id. Lets a
    // caller request `model: "code"` and have it resolve to the recommended
    // model before the registry lookup. Ships with built-in defaults (see
    // `build_model_aliases`); extend/override via `SIE_GATEWAY_MODEL_ALIASES`
    // (JSON map). Mirrors the `SIE_GATEWAY_GPU_ALIASES` mechanism.
    pub model_aliases: HashMap<String, String>,

    // Model registry paths (filesystem seed; same volume mounted into sie-config
    // for consistency, but the gateway never writes to them).
    pub bundles_dir: String,
    pub models_dir: String,

    // sie-config control plane URL. In-cluster Helm default is something like
    // `http://<release>-sie-config.<ns>.svc.cluster.local:8080`. When unset the
    // gateway runs without a bootstrap (useful in tests and single-process
    // examples); production Helm always sets this.
    pub config_service_url: Option<String>,

    // Admin token the gateway presents as a bearer credential when calling
    // `sie-config`'s bootstrap endpoints (`GET /v1/configs/export` and
    // `GET /v1/configs/epoch`). Reuses SIE_ADMIN_TOKEN because both services
    // share one admin secret in-cluster.
    pub config_service_token: Option<String>,

    // Optional Modal platform proxy-auth token the gateway's config client
    // presents ALONGSIDE `config_service_token` on `GET /v1/configs/*` calls
    // (superlinked/sie-internal #1740). `Some` only when both
    // `SIE_MODAL_PROXY_TOKEN_ID` and `SIE_MODAL_PROXY_TOKEN_SECRET` are set;
    // absent on self-host / dev (no Modal edge), so no proxy headers are sent.
    pub config_modal_proxy_token: Option<ModalProxyToken>,

    // Payload store (local path, s3://bucket/prefix, gs://bucket/prefix, or
    // abfs(s)://container@account.dfs.core.windows.net/prefix)
    pub payload_store_url: String,

    // The gateway's own EXTERNAL https origin (`SIE_GATEWAY_PUBLIC_URL`,
    // scheme+host, no trailing slash) — what absolute client-facing URLs are
    // minted against (e.g. the managed jobs API's §8 chunk-result capability
    // refs). `None` = unconfigured; callers derive the origin from the
    // request's Host headers instead.
    pub public_base_url: Option<String>,
}

/// Render a secret string for `Debug` output: an empty value stays empty (so
/// "unset" is still visible), any real value collapses to `<redacted>` so the
/// credential never reaches logs.
fn redacted_secret(value: &str) -> &'static str {
    if value.is_empty() {
        ""
    } else {
        "<redacted>"
    }
}

impl std::fmt::Debug for Config {
    /// Hand-written so credential-bearing fields never print their values.
    /// Non-secret fields stay fully visible for debuggability; the secret
    /// ones (`auth_tokens`, `admin_token`, `config_service_token`) redact while
    /// preserving present/absent (and count) so misconfig is still diagnosable.
    /// `config_modal_proxy_token` relies on `ModalProxyToken`'s own redacting
    /// `Debug`. `nats_url` / `payload_store_url` are connection URLs kept
    /// visible on purpose (redacting them would hide the target host).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Config")
            .field("host", &self.host)
            .field("port", &self.port)
            .field("worker_urls", &self.worker_urls)
            .field("use_kubernetes", &self.use_kubernetes)
            .field("k8s_namespace", &self.k8s_namespace)
            .field("k8s_service", &self.k8s_service)
            .field("k8s_port", &self.k8s_port)
            .field("health_mode", &self.health_mode)
            .field("nats_url", &self.nats_url)
            .field(
                "nats_config_trusted_producers",
                &self.nats_config_trusted_producers,
            )
            .field("auth_mode", &self.auth_mode)
            // Secret: the client API bearer tokens — show only how many.
            .field(
                "auth_tokens",
                &format_args!("<{} redacted>", self.auth_tokens.len()),
            )
            // Secret: the admin bearer (SIE_ADMIN_TOKEN).
            .field("admin_token", &redacted_secret(&self.admin_token))
            .field("auth_exempt_operational", &self.auth_exempt_operational)
            .field("log_level", &self.log_level)
            .field("json_logs", &self.json_logs)
            .field("enable_pools", &self.enable_pools)
            .field("hot_reload", &self.hot_reload)
            .field("watch_polling", &self.watch_polling)
            .field("multi_router", &self.multi_router)
            .field("request_timeout", &self.request_timeout)
            .field("max_stream_pending", &self.max_stream_pending)
            .field("stream_max_age_s", &self.stream_max_age_s)
            .field("configured_gpus", &self.configured_gpus)
            .field("gpu_profile_map", &self.gpu_profile_map)
            .field("configured_physical_lanes", &self.configured_physical_lanes)
            .field("static_queue_pools", &self.static_queue_pools)
            .field("model_aliases", &self.model_aliases)
            .field("bundles_dir", &self.bundles_dir)
            .field("models_dir", &self.models_dir)
            .field("config_service_url", &self.config_service_url)
            // Secret: same admin bearer as `admin_token`; keep present/absent.
            .field(
                "config_service_token",
                &self.config_service_token.as_ref().map(|_| "<redacted>"),
            )
            // `ModalProxyToken`'s own `Debug` redacts both halves.
            .field("config_modal_proxy_token", &self.config_modal_proxy_token)
            .field("payload_store_url", &self.payload_store_url)
            .field("public_base_url", &self.public_base_url)
            .finish()
    }
}

fn env_bool(key: &str) -> bool {
    match env::var(key) {
        Ok(v) => matches!(v.to_lowercase().as_str(), "true" | "1" | "yes"),
        Err(_) => false,
    }
}

/// Parse `raw` as an absolute HTTPS **origin** — `scheme://host[:port]` and
/// NOTHING else — returning the normalized origin (no trailing slash) or
/// `None`.
///
/// This is the ONE public-origin rule shared across the codebase (the cloud
/// gateway's request-origin derivation and the deploy-side validators mirror
/// it): HTTPS is required; plain `http` is allowed ONLY for loopback hosts
/// (`localhost` / `127.0.0.0/8` / `::1`) for local dev. A path, query string,
/// fragment, or userinfo is rejected — such an "origin" would mint broken or
/// credential-leaking capability URLs (the signature is bearer-equivalent, so
/// it must never ride a plaintext or malformed origin).
pub fn parse_https_origin(raw: &str) -> Option<String> {
    let raw = raw.trim();
    if raw.is_empty() || raw.contains(char::is_whitespace) {
        return None;
    }
    let (scheme, rest) = raw.split_once("://")?;
    let scheme = scheme.to_ascii_lowercase();
    if scheme != "https" && scheme != "http" {
        return None;
    }
    // Authority = up to the first `/`, `?`, or `#`. The remainder must be
    // empty or a single bare trailing `/` (the empty/root path, origin-
    // equivalent) — any real path, query, or fragment means this is not a bare
    // origin and is rejected.
    let authority_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let authority = &rest[..authority_end];
    let remainder = &rest[authority_end..];
    if !(remainder.is_empty() || remainder == "/") {
        return None;
    }
    // No userinfo.
    if authority.is_empty() || authority.contains('@') {
        return None;
    }
    let (host, port, is_ipv6) = split_host_port(authority)?;
    if host.is_empty() {
        return None;
    }
    let host_ok = if is_ipv6 {
        host.bytes()
            .all(|b| b.is_ascii_hexdigit() || b == b':' || b == b'.')
    } else {
        host.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'-')
    };
    if !host_ok {
        return None;
    }
    if let Some(port) = port {
        if port.is_empty() || !port.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        if port.parse::<u16>().is_err() {
            return None;
        }
    }
    if scheme == "http" && !host_is_loopback(host) {
        return None;
    }
    Some(format!("{scheme}://{authority}"))
}

/// Split an authority (no userinfo) into `(host, optional port, is_ipv6)`.
/// A bracketed `[..]` prefix marks an IPv6 literal (its host is the bracketed
/// content, port follows `]`); otherwise the last `:` separates an optional
/// port. `None` on a malformed bracket form.
fn split_host_port(authority: &str) -> Option<(&str, Option<&str>, bool)> {
    if let Some(rest) = authority.strip_prefix('[') {
        let (host, after) = rest.split_once(']')?;
        match after {
            "" => Some((host, None, true)),
            _ => Some((host, Some(after.strip_prefix(':')?), true)),
        }
    } else {
        match authority.rsplit_once(':') {
            Some((host, port)) => Some((host, Some(port), false)),
            None => Some((authority, None, false)),
        }
    }
}

/// Loopback host test for the `http`-allowed exception.
/// The ONE canonical loopback rule, shared by every capability-URL origin
/// check (this crate's `parse_https_origin` AND the cloud gateway's
/// request-origin derivation via [`authority_is_loopback`]): the exact
/// hostname `localhost` (case-insensitive), OR an IP LITERAL whose value is
/// loopback — `Ipv4Addr::is_loopback` (all of 127.0.0.0/8) or
/// `Ipv6Addr::is_loopback` (`::1` only; an IPv4-mapped `::ffff:127.0.0.1` is
/// NOT loopback, matching `IpAddr::is_loopback`).
///
/// NEVER a prefix match: `127.attacker.example` is a routable DNS name, so it
/// must not earn the plaintext-`http` exception (which would leak the
/// bearer-equivalent capability sig in transit). `host` is a bare host — no
/// port, IPv6 unbracketed.
pub fn host_is_loopback(host: &str) -> bool {
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    host.parse::<std::net::IpAddr>()
        .map(|a| a.is_loopback())
        .unwrap_or(false)
}

fn env_int(key: &str, fallback: u16) -> u16 {
    env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(fallback)
}

fn env_float(key: &str, fallback: f64) -> f64 {
    env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(fallback)
}

fn env_finite_float(key: &str, fallback: f64) -> f64 {
    let value = env_float(key, fallback);
    assert!(
        value.is_finite(),
        "{key} must be a finite floating point value"
    );
    value
}

fn env_u64(key: &str, fallback: u64) -> u64 {
    env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(fallback)
}

fn env_csv(key: &str) -> Vec<String> {
    match env::var(key) {
        Ok(s) if !s.is_empty() => s
            .split(',')
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .collect(),
        _ => Vec::new(),
    }
}

fn env_json_string_map(key: &str) -> HashMap<String, String> {
    match env::var(key) {
        Ok(s) if !s.trim().is_empty() => {
            serde_json::from_str::<HashMap<String, String>>(&s).unwrap_or_default()
        }
        _ => HashMap::new(),
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PhysicalLaneEnvSpec {
    pool: String,
    machine_profile: String,
    bundle: String,
}

fn env_physical_lane_catalog(key: &str) -> PhysicalLaneCatalog {
    let parsed = match env::var(key) {
        Ok(value) if !value.trim().is_empty() => {
            serde_json::from_str::<Vec<PhysicalLaneEnvSpec>>(&value).unwrap_or_else(|error| {
                panic!("failed to parse {key}: {error}; fix or unset {key}")
            })
        }
        _ => Vec::new(),
    };

    let lanes = parsed
        .into_iter()
        .map(|lane| (lane.pool, lane.machine_profile, lane.bundle));
    PhysicalLaneCatalog::try_from_raw(lanes)
        .unwrap_or_else(|error| panic!("invalid {key} lane: {error}; fix or unset {key}"))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct StaticQueuePoolEnvSpec {
    #[serde(default)]
    bundle: Option<String>,
    #[serde(default)]
    gpus: HashMap<String, u32>,
    #[serde(default, alias = "gpu_caps")]
    gpu_caps: HashMap<String, u32>,
    #[serde(default, alias = "minimum_worker_count")]
    minimum_worker_count: u32,
    #[serde(default, alias = "pinned_models")]
    pinned_models: Vec<String>,
}

fn env_static_queue_pools(key: &str) -> Vec<PoolSpec> {
    let parsed = match env::var(key) {
        Ok(s) if !s.trim().is_empty() => {
            match serde_json::from_str::<HashMap<String, StaticQueuePoolEnvSpec>>(&s) {
                Ok(parsed) => parsed,
                Err(error) => {
                    panic!("failed to parse {key}: {error}; fix or unset {key}")
                }
            }
        }
        _ => HashMap::new(),
    };

    let mut specs: Vec<PoolSpec> = parsed
        .into_iter()
        .filter_map(|(name, spec)| {
            let name = name.trim().to_string();
            if name.is_empty() {
                return None;
            }
            let bundle = spec.bundle.and_then(|bundle| {
                let bundle = bundle.trim().to_string();
                if bundle.is_empty() {
                    None
                } else {
                    Some(bundle)
                }
            });
            Some(PoolSpec {
                queue_pool: name.clone(),
                name,
                bundle,
                gpus: spec.gpus,
                gpu_caps: spec.gpu_caps,
                ttl_seconds: None,
                minimum_worker_count: spec.minimum_worker_count,
                // Static-pool pinned sets are intentionally NOT registry-validated:
                // they are parsed at startup before the model registry is populated.
                pinned_models: spec.pinned_models,
            })
        })
        .collect();
    specs.sort_by(|a, b| a.name.cmp(&b.name));
    specs
}

/// Job/friendly model aliases: lowercase alias → model id (or `bundle:/model`).
///
/// Ships with built-in defaults so `model="code"` works without operator
/// config (and makes "each agent job routes to the right model" true by
/// default). `SIE_GATEWAY_MODEL_ALIASES` (a JSON map) extends or overrides
/// them. Empty alias/target pairs are skipped; aliases are stored lowercased
/// because resolution lowercases the lookup.
///
/// A target may be a bare model id (`Org/Model`), a concrete profile variant
/// (`Org/Model:profile`), or a bundle-qualified spec (`bundle:/Org/Model`).
/// The bundle form lets an operator pin a precision / profile bundle for a job
/// — e.g. map `sql` to a BF16 bundle that avoids the FP8 SQL-accuracy
/// regression (ADR 0001). `resolve_model_spec_with_aliases` (proxy.rs) applies
/// the bundle and preserves concrete profile variants.
fn build_model_aliases(overrides: HashMap<String, String>) -> HashMap<String, String> {
    let mut map: HashMap<String, String> = HashMap::new();
    // Built-in: the code-generation job → the model with a MEASURED
    // HumanEval/MBPP pass@1 baseline that also serves reliably
    // (Qwen3-4B-Instruct-2507: 0.866 / 0.74). Qwen3.5-4B is stronger on paper
    // but its NEXTN/hybrid serving path does not come up reliably yet, so it is
    // not the default until measured + its serving init is fixed.
    map.insert(
        "code".to_string(),
        "Qwen/Qwen3-4B-Instruct-2507".to_string(),
    );
    // Text-to-SQL job → an ebnf-grammar-capable LLM for the "any LLM + SQL
    // grammar" path. Repoint to SQLCoder once that model is onboarded.
    map.insert("sql".to_string(), "Qwen/Qwen3-4B-Instruct-2507".to_string());
    // CHECK POLICY job → a generative guard model that emits a safe/unsafe
    // verdict. Granite Guardian 3.0 2B (Apache-2.0, ungated) measured on
    // ToxicChat via the generation gate; serves on the same SGLang path.
    map.insert(
        "guard".to_string(),
        "ibm-granite/granite-guardian-3.0-2b".to_string(),
    );
    for (alias, target) in overrides {
        let alias = alias.trim().to_lowercase();
        let target = target.trim().to_string();
        if alias.is_empty() || target.is_empty() {
            continue;
        }
        map.insert(alias, target);
    }
    map
}

fn build_gpu_profile_map(
    configured_gpus: &[String],
    aliases: HashMap<String, String>,
) -> HashMap<String, String> {
    let mut map: HashMap<String, String> = configured_gpus
        .iter()
        .map(|g| (g.to_lowercase(), g.clone()))
        .collect();

    for (alias, profile) in aliases {
        let alias = alias.trim();
        let profile = profile.trim();
        if alias.is_empty() || profile.is_empty() {
            continue;
        }
        map.entry(alias.to_lowercase())
            .or_insert_with(|| profile.to_string());
    }

    map
}

fn env_default(key: &str, fallback: &str) -> String {
    match env::var(key) {
        Ok(v) if !v.is_empty() => v,
        _ => fallback.to_string(),
    }
}

impl Config {
    pub fn load() -> Self {
        let mut auth_tokens = env_csv("SIE_AUTH_TOKENS");
        if auth_tokens.is_empty() {
            auth_tokens = env_csv("SIE_AUTH_TOKEN");
        }
        let configured_gpus = env_csv("SIE_GATEWAY_CONFIGURED_GPUS");
        let configured_physical_lanes =
            env_physical_lane_catalog("SIE_GATEWAY_CONFIGURED_PHYSICAL_LANES");
        let gpu_profile_map = build_gpu_profile_map(
            &configured_gpus,
            env_json_string_map("SIE_GATEWAY_GPU_ALIASES"),
        );
        let model_aliases = build_model_aliases(env_json_string_map("SIE_GATEWAY_MODEL_ALIASES"));

        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,

            worker_urls: env_csv("SIE_GATEWAY_WORKERS"),
            use_kubernetes: env_bool("SIE_GATEWAY_KUBERNETES"),
            k8s_namespace: env_default("SIE_GATEWAY_K8S_NAMESPACE", "default"),
            k8s_service: env_default("SIE_GATEWAY_K8S_SERVICE", "sie-worker"),
            k8s_port: env_int("SIE_GATEWAY_K8S_PORT", 8080),

            health_mode: env_default("SIE_GATEWAY_HEALTH_MODE", "ws"),

            nats_url: env::var("SIE_NATS_URL").unwrap_or_default(),
            nats_config_trusted_producers: {
                // Explicit opt-in to the legacy "trust anyone" behavior.
                if env_bool("SIE_NATS_CONFIG_TRUST_ANY_PRODUCER") {
                    Vec::new()
                } else {
                    let custom = env_csv("SIE_NATS_CONFIG_TRUSTED_PRODUCERS");
                    if custom.is_empty() {
                        vec!["sie-config".to_string()]
                    } else {
                        custom
                    }
                }
            },

            auth_mode: env_default("SIE_AUTH_MODE", "none"),
            auth_tokens,
            admin_token: env::var("SIE_ADMIN_TOKEN").unwrap_or_default(),
            auth_exempt_operational: env_bool("SIE_AUTH_EXEMPT_OPERATIONAL"),

            log_level: env_default("SIE_LOG_LEVEL", "info"),
            json_logs: env_bool("SIE_LOG_JSON"),

            enable_pools: env_bool("SIE_GATEWAY_ENABLE_POOLS"),
            hot_reload: env_bool("SIE_GATEWAY_HOT_RELOAD"),
            watch_polling: env_bool("SIE_GATEWAY_WATCH_POLLING")
                || env_bool("SIE_GATEWAY_POLLING_WATCHER"),
            multi_router: env_bool("SIE_MULTI_ROUTER"),

            request_timeout: env_finite_float("SIE_GATEWAY_REQUEST_TIMEOUT", 120.0),
            max_stream_pending: env_u64("SIE_GATEWAY_MAX_STREAM_PENDING", 50_000),
            stream_max_age_s: env_u64("SIE_STREAM_MAX_AGE_S", 1_800),

            configured_gpus,
            gpu_profile_map,
            configured_physical_lanes,
            static_queue_pools: env_static_queue_pools("SIE_GATEWAY_STATIC_QUEUE_POOLS"),
            model_aliases,

            bundles_dir: env_default("SIE_BUNDLES_DIR", "bundles"),
            models_dir: env_default("SIE_MODELS_DIR", "models"),

            config_service_url: {
                let raw = env::var("SIE_CONFIG_SERVICE_URL").unwrap_or_default();
                if raw.is_empty() {
                    None
                } else {
                    Some(raw)
                }
            },
            config_service_token: {
                let raw = env::var("SIE_ADMIN_TOKEN").unwrap_or_default();
                if raw.is_empty() {
                    None
                } else {
                    Some(raw)
                }
            },
            config_modal_proxy_token: {
                // #1740: opt-in Modal platform proxy-auth. Both halves required
                // for the pair to be sent — a half-set pair is a misconfig, not
                // partial auth, so it degrades to "not configured" (headers
                // omitted) rather than sending a broken credential.
                let key = env::var("SIE_MODAL_PROXY_TOKEN_ID").unwrap_or_default();
                let secret = env::var("SIE_MODAL_PROXY_TOKEN_SECRET").unwrap_or_default();
                if key.is_empty() || secret.is_empty() {
                    None
                } else {
                    Some(ModalProxyToken { key, secret })
                }
            },

            payload_store_url: env_default("SIE_PAYLOAD_STORE_URL", ""),

            // Validated fail-closed: only an exact HTTPS origin is accepted
            // (see `parse_https_origin`). An unset value, or one carrying a
            // path/query/fragment/userinfo or a non-loopback `http://` scheme,
            // resolves to `None` (the minter then derives the origin from the
            // request's Host headers). A capability signature is bearer-
            // equivalent, so a plaintext origin must never mint it.
            public_base_url: {
                let raw = env_default("SIE_GATEWAY_PUBLIC_URL", "");
                let raw = raw.trim();
                if raw.is_empty() {
                    None
                } else {
                    let origin = parse_https_origin(raw);
                    if origin.is_none() {
                        tracing::warn!(
                            "SIE_GATEWAY_PUBLIC_URL is not a bare HTTPS origin \
                             (scheme+host[:port] only, https outside loopback) — ignoring it; \
                             client-facing URLs will be derived from request Host headers"
                        );
                    }
                    origin
                }
            },
        }
    }

    /// Report auth configuration soundness. Returns `(level, message)`
    /// pairs that callers log at startup. Catches fail-open
    /// misconfigurations (e.g. tokens set while `SIE_AUTH_MODE=none`),
    /// unknown modes, missing tokens, and explicit operational bypasses.
    ///
    /// Does not mutate `self` and does not refuse startup; matches the
    /// gateway's "log and continue" posture for config issues.
    pub fn audit_auth(&self) -> Vec<(AuditLevel, String)> {
        let mut issues = Vec::new();
        let mode = self.auth_mode.as_str();
        let has_tokens = !self.auth_tokens.is_empty();
        let has_admin = !self.admin_token.is_empty();

        let is_enabled = matches!(mode, "static" | "token");
        let is_disabled = matches!(mode, "none" | "");

        if !is_enabled && !is_disabled {
            issues.push((
                AuditLevel::Error,
                format!(
                    "SIE_AUTH_MODE='{}' is not recognized; expected 'none', 'static', or 'token'. Auth is currently DISABLED (fail-open) because of the unknown mode — fix SIE_AUTH_MODE.",
                    mode
                ),
            ));
        }

        if !is_enabled && (has_tokens || has_admin) {
            issues.push((
                AuditLevel::Error,
                "SIE_AUTH_TOKEN(S) or SIE_ADMIN_TOKEN is set but SIE_AUTH_MODE is not 'static'/'token'. Auth is DISABLED; the tokens are dead configuration. Set SIE_AUTH_MODE=token to enforce auth.".to_string(),
            ));
        }

        if is_enabled && !has_tokens {
            issues.push((
                AuditLevel::Error,
                "Auth is enabled but SIE_AUTH_TOKEN(S) is empty. All non-probe requests will be rejected with 500.".to_string(),
            ));
        }

        if is_enabled && !has_admin {
            issues.push((
                AuditLevel::Warn,
                "Auth is enabled but SIE_ADMIN_TOKEN is unset. Admin-only endpoints (config writes, pool mutations) will refuse with 403 until an admin token is configured.".to_string(),
            ));
        }

        if is_enabled && self.auth_exempt_operational {
            issues.push((
                AuditLevel::Warn,
                "SIE_AUTH_EXEMPT_OPERATIONAL=true: status page, /health, and /ws/* bypass auth. Use only when those endpoints are already network-isolated.".to_string(),
            ));
        }

        issues
    }

    /// Report NATS config-delta producer-trust soundness. Mirrors the
    /// pattern of `audit_auth` but scoped to the
    /// `SIE_NATS_CONFIG_TRUST_ANY_PRODUCER` /
    /// `SIE_NATS_CONFIG_TRUSTED_PRODUCERS` pair. Emitted at startup.
    pub fn audit_nats_producer_trust(&self) -> Vec<(AuditLevel, String)> {
        let mut issues = Vec::new();
        // Both flags cannot be observed independently from `self` because
        // the load step collapses them into a single `Vec<String>`. We
        // detect the conflict by re-reading the env: "trust any" wins on
        // collapse, so if the allowlist env is *also* set we warn that it
        // is silently ignored. This is cheap and only runs once at boot.
        let trust_any = env_bool("SIE_NATS_CONFIG_TRUST_ANY_PRODUCER");
        let has_custom_allowlist = !env_csv("SIE_NATS_CONFIG_TRUSTED_PRODUCERS").is_empty();
        if trust_any && has_custom_allowlist {
            issues.push((
                AuditLevel::Warn,
                "SIE_NATS_CONFIG_TRUST_ANY_PRODUCER=true overrides SIE_NATS_CONFIG_TRUSTED_PRODUCERS; the allowlist is ignored. Unset one.".to_string(),
            ));
        }
        if self.nats_config_trusted_producers.is_empty() {
            issues.push((
                AuditLevel::Warn,
                "NATS config-delta producer validation is DISABLED; any publisher on sie.config.models._all will be accepted. Intended for local dev / single-node test clusters.".to_string(),
            ));
        }
        issues
    }
}

/// Severity for a `Config::audit_auth` finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditLevel {
    Warn,
    Error,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Serialize env-var tests to avoid races (env vars are process-global).
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn with_env<F: FnOnce()>(vars: &[(&str, &str)], f: F) {
        let guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let old: Vec<(&str, Option<String>)> =
            vars.iter().map(|(k, _)| (*k, env::var(k).ok())).collect();
        for (k, v) in vars {
            env::set_var(k, v);
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        for (k, old_val) in old {
            match old_val {
                Some(v) => env::set_var(k, v),
                None => env::remove_var(k),
            }
        }
        drop(guard);
        if let Err(panic) = result {
            std::panic::resume_unwind(panic);
        }
    }

    fn without_env<F: FnOnce()>(keys: &[&str], f: F) {
        let guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let old: Vec<(&str, Option<String>)> =
            keys.iter().map(|k| (*k, env::var(k).ok())).collect();
        for k in keys {
            env::remove_var(k);
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        for (k, old_val) in old {
            match old_val {
                Some(v) => env::set_var(k, v),
                None => env::remove_var(k),
            }
        }
        drop(guard);
        if let Err(panic) = result {
            std::panic::resume_unwind(panic);
        }
    }

    // ── env_bool ───────────────────────────────────────────────────

    #[test]
    fn test_env_bool_true_values() {
        for val in &["true", "1", "yes", "True", "YES", "TRUE"] {
            with_env(&[("_TEST_BOOL", val)], || {
                assert!(env_bool("_TEST_BOOL"), "expected true for '{}'", val);
            });
        }
    }

    #[test]
    fn test_env_bool_false_values() {
        for val in &["false", "0", "no", "anything"] {
            with_env(&[("_TEST_BOOL", val)], || {
                assert!(!env_bool("_TEST_BOOL"), "expected false for '{}'", val);
            });
        }
    }

    #[test]
    fn test_env_bool_missing() {
        without_env(&["_TEST_BOOL_MISSING"], || {
            assert!(!env_bool("_TEST_BOOL_MISSING"));
        });
    }

    // ── env_int ────────────────────────────────────────────────────

    #[test]
    fn test_env_int_valid() {
        with_env(&[("_TEST_INT", "9090")], || {
            assert_eq!(env_int("_TEST_INT", 80), 9090);
        });
    }

    #[test]
    fn test_env_int_invalid_uses_fallback() {
        with_env(&[("_TEST_INT", "not_a_number")], || {
            assert_eq!(env_int("_TEST_INT", 80), 80);
        });
    }

    #[test]
    fn test_env_int_missing_uses_fallback() {
        without_env(&["_TEST_INT_MISSING"], || {
            assert_eq!(env_int("_TEST_INT_MISSING", 8080), 8080);
        });
    }

    // ── env_float ──────────────────────────────────────────────────

    #[test]
    fn test_env_float_valid() {
        with_env(&[("_TEST_FLOAT", "2.5")], || {
            assert!((env_float("_TEST_FLOAT", 1.0) - 2.5).abs() < f64::EPSILON);
        });
    }

    #[test]
    fn test_env_float_fallback() {
        without_env(&["_TEST_FLOAT_MISSING"], || {
            assert!((env_float("_TEST_FLOAT_MISSING", 30.0) - 30.0).abs() < f64::EPSILON);
        });
    }

    // ── env_csv ────────────────────────────────────────────────────

    #[test]
    fn test_env_csv_multiple() {
        with_env(
            &[("_TEST_CSV", "http://a:80, http://b:80, http://c:80")],
            || {
                let result = env_csv("_TEST_CSV");
                assert_eq!(result, vec!["http://a:80", "http://b:80", "http://c:80"]);
            },
        );
    }

    #[test]
    fn test_env_csv_empty() {
        with_env(&[("_TEST_CSV", "")], || {
            assert!(env_csv("_TEST_CSV").is_empty());
        });
    }

    #[test]
    fn test_env_csv_missing() {
        without_env(&["_TEST_CSV_MISSING"], || {
            assert!(env_csv("_TEST_CSV_MISSING").is_empty());
        });
    }

    #[test]
    fn test_env_csv_trims_whitespace() {
        with_env(&[("_TEST_CSV", "  a , b , c  ")], || {
            assert_eq!(env_csv("_TEST_CSV"), vec!["a", "b", "c"]);
        });
    }

    #[test]
    fn test_env_csv_filters_empty_entries() {
        with_env(&[("_TEST_CSV", "a,,b,")], || {
            assert_eq!(env_csv("_TEST_CSV"), vec!["a", "b"]);
        });
    }

    #[test]
    fn test_env_json_string_map() {
        with_env(&[("_TEST_JSON_MAP", r#"{"l4":"l4-spot"}"#)], || {
            let result = env_json_string_map("_TEST_JSON_MAP");
            assert_eq!(result.get("l4"), Some(&"l4-spot".to_string()));
        });
    }

    #[test]
    fn test_env_json_string_map_invalid_is_empty() {
        with_env(&[("_TEST_JSON_MAP", "not-json")], || {
            assert!(env_json_string_map("_TEST_JSON_MAP").is_empty());
        });
    }

    #[test]
    fn test_env_physical_lane_catalog_resolves_exact_canonical_tuple() {
        with_env(
            &[(
                "_TEST_PHYSICAL_LANES",
                r#"[{"pool":" Customer_A ","machineProfile":" CPU_One ","bundle":" DEFAULT "}]"#,
            )],
            || {
                let catalog = env_physical_lane_catalog("_TEST_PHYSICAL_LANES");
                let lane = catalog
                    .resolve("customer_a", "cpu_one", "default")
                    .expect("rendered KEDA lane should resolve");
                assert_eq!(lane.to_string(), "customer_a|cpu_one|default");
                assert!(catalog
                    .resolve("customer_a", "caller-profile", "default")
                    .is_none());
            },
        );
    }

    #[test]
    fn test_env_physical_lane_catalog_invalid_input_panics() {
        with_env(
            &[(
                "_TEST_PHYSICAL_LANES",
                r#"[{"pool":"default","machineProfile":"l4/spot","bundle":"default"}]"#,
            )],
            || {
                let result = std::panic::catch_unwind(|| {
                    let _ = env_physical_lane_catalog("_TEST_PHYSICAL_LANES");
                });
                let panic = result.expect_err("invalid KEDA lane catalog must fail fast");
                let message = panic
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| panic.downcast_ref::<&str>().copied())
                    .unwrap_or("");
                assert!(message.contains("invalid _TEST_PHYSICAL_LANES lane"));
            },
        );
    }

    #[test]
    fn test_env_static_queue_pools_accepts_helm_shape() {
        with_env(
            &[(
                "_TEST_STATIC_QUEUE_POOLS",
                r#"{
                    "companyA": {
                        "gpus": {"l4": 0},
                        "gpuCaps": {},
                        "minimumWorkerCount": 1,
                        "pinnedModels": ["BAAI/bge-m3"]
                    },
                    "companyB": {
                        "bundle": "sglang",
                        "gpus": {"a100": 1},
                        "gpu_caps": {"a100": 2},
                        "minimum_worker_count": 2,
                        "pinned_models": ["intfloat/e5-base-v2"]
                    }
                }"#,
            )],
            || {
                let result = env_static_queue_pools("_TEST_STATIC_QUEUE_POOLS");

                assert_eq!(result.len(), 2);
                assert_eq!(result[0].name, "companyA");
                assert_eq!(result[0].gpus.get("l4"), Some(&0));
                assert!(result[0].gpu_caps.is_empty());
                assert_eq!(result[0].minimum_worker_count, 1);
                assert_eq!(result[0].pinned_models, vec!["BAAI/bge-m3"]);
                assert_eq!(result[0].ttl_seconds, None);

                assert_eq!(result[1].name, "companyB");
                assert_eq!(result[1].bundle.as_deref(), Some("sglang"));
                assert_eq!(result[1].gpus.get("a100"), Some(&1));
                assert_eq!(result[1].gpu_caps.get("a100"), Some(&2));
                assert_eq!(result[1].minimum_worker_count, 2);
                assert_eq!(result[1].pinned_models, vec!["intfloat/e5-base-v2"]);
            },
        );
    }

    #[test]
    fn test_env_static_queue_pools_minimum_worker_count_defaults_to_zero() {
        with_env(
            &[(
                "_TEST_STATIC_QUEUE_POOLS",
                r#"{"companyC": {"gpus": {"l4": 0}}}"#,
            )],
            || {
                let result = env_static_queue_pools("_TEST_STATIC_QUEUE_POOLS");
                assert_eq!(result.len(), 1);
                assert_eq!(result[0].minimum_worker_count, 0);
                assert!(result[0].pinned_models.is_empty());
            },
        );
    }

    #[test]
    fn test_env_static_queue_pools_invalid_panics() {
        with_env(&[("_TEST_STATIC_QUEUE_POOLS", "not-json")], || {
            let result = std::panic::catch_unwind(|| {
                let _ = env_static_queue_pools("_TEST_STATIC_QUEUE_POOLS");
            });
            let panic = result.expect_err("invalid static queue pool JSON must fail fast");
            let message = panic
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .unwrap_or("");
            assert!(message.contains("failed to parse _TEST_STATIC_QUEUE_POOLS"));
        });
    }

    #[test]
    fn test_build_gpu_profile_map_preserves_canonical_and_aliases() {
        let mut aliases = HashMap::new();
        aliases.insert("l4".to_string(), "l4-spot".to_string());

        let result = build_gpu_profile_map(&["l4-spot".to_string()], aliases);

        assert_eq!(result.get("l4-spot"), Some(&"l4-spot".to_string()));
        assert_eq!(result.get("l4"), Some(&"l4-spot".to_string()));
    }

    #[test]
    fn test_build_model_aliases_has_builtin_code_default() {
        let result = build_model_aliases(HashMap::new());
        assert_eq!(
            result.get("code"),
            Some(&"Qwen/Qwen3-4B-Instruct-2507".to_string())
        );
        assert_eq!(
            result.get("sql"),
            Some(&"Qwen/Qwen3-4B-Instruct-2507".to_string())
        );
        assert_eq!(
            result.get("guard"),
            Some(&"ibm-granite/granite-guardian-3.0-2b".to_string())
        );
    }

    #[test]
    fn test_build_model_aliases_env_overrides_and_extends() {
        let mut overrides = HashMap::new();
        overrides.insert("code".to_string(), "Org/Coder".to_string()); // override default
        overrides.insert("SQL".to_string(), "Org/SQLModel".to_string()); // extend + lowercased
        overrides.insert("blank".to_string(), "".to_string()); // skipped (empty target)

        let result = build_model_aliases(overrides);

        assert_eq!(result.get("code"), Some(&"Org/Coder".to_string()));
        assert_eq!(result.get("sql"), Some(&"Org/SQLModel".to_string()));
        assert!(!result.contains_key("blank"));
    }

    #[test]
    fn test_build_gpu_profile_map_does_not_override_canonical_profile() {
        let mut aliases = HashMap::new();
        aliases.insert("l4".to_string(), "l4-spot".to_string());

        let result = build_gpu_profile_map(&["l4".to_string(), "l4-spot".to_string()], aliases);

        assert_eq!(result.get("l4"), Some(&"l4".to_string()));
    }

    #[test]
    fn test_config_load_uses_gpu_aliases() {
        with_env(
            &[
                ("SIE_GATEWAY_CONFIGURED_GPUS", "l4-spot"),
                ("SIE_GATEWAY_GPU_ALIASES", r#"{"l4":"l4-spot"}"#),
            ],
            || {
                let cfg = Config::load();
                assert_eq!(cfg.configured_gpus, vec!["l4-spot"]);
                assert_eq!(cfg.gpu_profile_map.get("l4"), Some(&"l4-spot".to_string()));
            },
        );
    }

    // ── env_default ────────────────────────────────────────────────

    #[test]
    fn test_env_default_set() {
        with_env(&[("_TEST_DEFAULT", "custom_value")], || {
            assert_eq!(env_default("_TEST_DEFAULT", "fallback"), "custom_value");
        });
    }

    #[test]
    fn test_env_default_empty_uses_fallback() {
        with_env(&[("_TEST_DEFAULT", "")], || {
            assert_eq!(env_default("_TEST_DEFAULT", "fallback"), "fallback");
        });
    }

    #[test]
    fn test_env_default_missing_uses_fallback() {
        without_env(&["_TEST_DEFAULT_MISSING"], || {
            assert_eq!(env_default("_TEST_DEFAULT_MISSING", "fallback"), "fallback");
        });
    }

    // ── env_u64, env_usize ─────────────────────────────────────────

    #[test]
    fn test_env_u64() {
        with_env(&[("_TEST_U64", "12345")], || {
            assert_eq!(env_u64("_TEST_U64", 0), 12345);
        });
    }

    // ── Config.load integration ───────────────────────────────────

    #[test]
    fn test_config_service_url_unset_is_none() {
        without_env(&["SIE_CONFIG_SERVICE_URL"], || {
            let cfg = Config::load();
            assert!(cfg.config_service_url.is_none());
        });
    }

    #[test]
    fn test_request_timeout_default_is_120_seconds() {
        without_env(&["SIE_GATEWAY_REQUEST_TIMEOUT"], || {
            let cfg = Config::load();
            assert!((cfg.request_timeout - 120.0).abs() < f64::EPSILON);
        });
    }

    #[test]
    fn test_request_timeout_from_env() {
        with_env(&[("SIE_GATEWAY_REQUEST_TIMEOUT", "45.5")], || {
            let cfg = Config::load();
            assert!((cfg.request_timeout - 45.5).abs() < f64::EPSILON);
        });
    }

    #[test]
    #[should_panic(expected = "SIE_GATEWAY_REQUEST_TIMEOUT must be a finite floating point value")]
    fn test_request_timeout_rejects_non_finite_env() {
        with_env(&[("SIE_GATEWAY_REQUEST_TIMEOUT", "NaN")], || {
            let _ = Config::load();
        });
    }

    #[test]
    fn test_config_service_url_from_env() {
        with_env(
            &[(
                "SIE_CONFIG_SERVICE_URL",
                "http://sie-config.sie.svc.cluster.local:8080",
            )],
            || {
                let cfg = Config::load();
                assert_eq!(
                    cfg.config_service_url.as_deref(),
                    Some("http://sie-config.sie.svc.cluster.local:8080"),
                );
            },
        );
    }

    #[test]
    fn test_config_service_url_empty_is_none() {
        with_env(&[("SIE_CONFIG_SERVICE_URL", "")], || {
            let cfg = Config::load();
            assert!(cfg.config_service_url.is_none());
        });
    }

    #[test]
    fn test_payload_store_url_default() {
        without_env(&["SIE_PAYLOAD_STORE_URL"], || {
            let cfg = Config::load();
            assert_eq!(cfg.payload_store_url, "");
        });
    }

    #[test]
    fn test_payload_store_url_from_env() {
        with_env(
            &[("SIE_PAYLOAD_STORE_URL", "s3://my-bucket/payloads")],
            || {
                let cfg = Config::load();
                assert_eq!(cfg.payload_store_url, "s3://my-bucket/payloads");
            },
        );
    }

    #[test]
    fn test_public_base_url_default_is_none() {
        without_env(&["SIE_GATEWAY_PUBLIC_URL"], || {
            let cfg = Config::load();
            assert!(cfg.public_base_url.is_none());
        });
    }

    #[test]
    fn test_public_base_url_trims_trailing_slash() {
        with_env(
            &[("SIE_GATEWAY_PUBLIC_URL", "https://api.example.com/")],
            || {
                let cfg = Config::load();
                assert_eq!(
                    cfg.public_base_url.as_deref(),
                    Some("https://api.example.com")
                );
            },
        );
    }

    #[test]
    fn test_public_base_url_empty_is_none() {
        with_env(&[("SIE_GATEWAY_PUBLIC_URL", "")], || {
            let cfg = Config::load();
            assert!(cfg.public_base_url.is_none());
        });
    }

    #[test]
    fn test_public_base_url_rejects_invalid_origins() {
        // Fail closed on anything that is not a bare HTTPS origin: a capability
        // signature is bearer-equivalent, so a plaintext/path-bearing/userinfo
        // origin must never mint it (CodeRabbit #1).
        for bad in [
            "http://api.example.com",            // non-loopback http
            "http://127.attacker.example",       // prefix-bypass: routable DNS, NOT loopback
            "http://127.0.0.1.attacker.example", // ditto
            "https://api.example.com/v1",        // path
            "https://api.example.com?x=1",       // query
            "https://api.example.com#f",         // fragment
            "https://user:pass@api.example.com", // userinfo
            "https://user@api.example.com",      // userinfo
            "ftp://api.example.com",             // wrong scheme
            "api.example.com",                   // no scheme
            "https://",                          // no host
            "https://api example.com",           // whitespace
            "https://api.example.com:notaport",  // bad port
        ] {
            with_env(&[("SIE_GATEWAY_PUBLIC_URL", bad)], || {
                assert!(
                    Config::load().public_base_url.is_none(),
                    "expected {bad:?} to be rejected"
                );
            });
        }
    }

    #[test]
    fn test_public_base_url_accepts_valid_origins() {
        for (raw, want) in [
            ("https://api.example.com", "https://api.example.com"),
            (
                "https://api.example.com:8443",
                "https://api.example.com:8443",
            ),
            ("http://localhost:8080", "http://localhost:8080"),
            ("http://127.0.0.1:8080", "http://127.0.0.1:8080"),
            ("HTTPS://api.example.com", "https://api.example.com"),
        ] {
            with_env(&[("SIE_GATEWAY_PUBLIC_URL", raw)], || {
                assert_eq!(
                    Config::load().public_base_url.as_deref(),
                    Some(want),
                    "{raw}"
                );
            });
        }
    }

    #[test]
    fn test_parse_https_origin_unit() {
        assert_eq!(
            parse_https_origin("https://a.b/"),
            Some("https://a.b".to_string())
        );
        assert_eq!(
            parse_https_origin("http://[::1]:9"),
            Some("http://[::1]:9".to_string())
        );
        assert_eq!(
            parse_https_origin("http://[2001:db8::1]"),
            None,
            "non-loopback ipv6 http"
        );
        assert_eq!(
            parse_https_origin("https://[2001:db8::1]:443"),
            Some("https://[2001:db8::1]:443".to_string())
        );
        assert_eq!(parse_https_origin("https://a.b/x"), None);
        assert_eq!(parse_https_origin("https://a.b#"), None);
    }

    #[test]
    fn test_parse_https_origin_loopback_is_ip_literal_not_prefix() {
        // The http exception is IP-literal-loopback or exact `localhost` ONLY.
        assert_eq!(
            parse_https_origin("http://127.0.0.1"),
            Some("http://127.0.0.1".to_string())
        );
        assert_eq!(
            parse_https_origin("http://127.5.5.5:9"),
            Some("http://127.5.5.5:9".to_string()),
            "the whole 127.0.0.0/8 block is loopback"
        );
        assert_eq!(
            parse_https_origin("http://localhost"),
            Some("http://localhost".to_string())
        );
        // localhost is case-insensitive (parity with the Python mirror, which
        // gets a lowercased host from urlsplit).
        assert_eq!(
            parse_https_origin("http://LOCALHOST"),
            Some("http://LOCALHOST".to_string())
        );
        // An IPv4-mapped IPv6 literal is NOT loopback (IpAddr::is_loopback does
        // not unwrap the mapping — only ::1 / 127.0.0.0/8 count).
        assert_eq!(parse_https_origin("http://[::ffff:127.0.0.1]"), None);
        // Prefix-bypass hosts are routable DNS names — must be rejected.
        assert_eq!(parse_https_origin("http://127.attacker.example"), None);
        assert_eq!(
            parse_https_origin("http://127.0.0.1.attacker.example"),
            None
        );
        assert_eq!(
            parse_https_origin("http://localhost.attacker.example"),
            None
        );
        // https to those hosts is still fine (scheme is safe).
        assert_eq!(
            parse_https_origin("https://127.attacker.example"),
            Some("https://127.attacker.example".to_string())
        );
    }

    #[test]
    fn test_host_is_loopback_canonical() {
        assert!(host_is_loopback("localhost"));
        assert!(host_is_loopback("LocalHost"));
        assert!(host_is_loopback("127.0.0.1"));
        assert!(host_is_loopback("127.5.5.5")); // whole 127.0.0.0/8
        assert!(host_is_loopback("::1"));
        assert!(!host_is_loopback("::ffff:127.0.0.1")); // mapped v6: not ::1
        assert!(!host_is_loopback("127.attacker.example"));
        assert!(!host_is_loopback("10.0.0.1"));
    }

    #[test]
    fn test_stream_max_age_default_matches_worker_contract() {
        without_env(&["SIE_STREAM_MAX_AGE_S"], || {
            let cfg = Config::load();
            assert_eq!(cfg.stream_max_age_s, 1_800);
        });
    }

    #[test]
    fn test_stream_max_age_from_env() {
        with_env(&[("SIE_STREAM_MAX_AGE_S", "240")], || {
            let cfg = Config::load();
            assert_eq!(cfg.stream_max_age_s, 240);
        });
    }

    #[test]
    fn test_admin_token_populates_config_service_token() {
        with_env(&[("SIE_ADMIN_TOKEN", "super-secret")], || {
            let cfg = Config::load();
            assert_eq!(cfg.admin_token, "super-secret");
            assert_eq!(cfg.config_service_token.as_deref(), Some("super-secret"));
        });
    }

    #[test]
    fn test_admin_token_unset_leaves_config_service_token_none() {
        without_env(&["SIE_ADMIN_TOKEN"], || {
            let cfg = Config::load();
            assert!(cfg.admin_token.is_empty());
            assert!(cfg.config_service_token.is_none());
        });
    }

    #[test]
    fn test_modal_proxy_token_requires_both_halves() {
        // #1740: both env vars must be set for the pair to be presented.
        with_env(
            &[
                ("SIE_MODAL_PROXY_TOKEN_ID", "wk-abc"),
                ("SIE_MODAL_PROXY_TOKEN_SECRET", "ws-xyz"),
            ],
            || {
                let cfg = Config::load();
                let tok = cfg
                    .config_modal_proxy_token
                    .expect("both halves set -> Some");
                assert_eq!(tok.key, "wk-abc");
                assert_eq!(tok.secret, "ws-xyz");
            },
        );
    }

    #[test]
    fn test_modal_proxy_token_half_set_is_none() {
        // A half-set pair is a misconfig, not partial auth: degrade to None so
        // no broken credential is ever sent. An empty half is treated as unset
        // (the load step reads env::var(..).unwrap_or_default()).
        with_env(
            &[
                ("SIE_MODAL_PROXY_TOKEN_ID", "wk-abc"),
                ("SIE_MODAL_PROXY_TOKEN_SECRET", ""),
            ],
            || {
                assert!(Config::load().config_modal_proxy_token.is_none());
            },
        );
    }

    #[test]
    fn test_modal_proxy_token_unset_is_none() {
        without_env(
            &["SIE_MODAL_PROXY_TOKEN_ID", "SIE_MODAL_PROXY_TOKEN_SECRET"],
            || {
                assert!(Config::load().config_modal_proxy_token.is_none());
            },
        );
    }

    #[test]
    fn test_modal_proxy_token_debug_redacts_secret() {
        // `Config` is `{:?}`-formatted at startup and in tests, so the token's
        // manual Debug impl MUST NOT leak either half into logs.
        let token = ModalProxyToken {
            key: "wk-super-secret-id".to_string(),
            secret: "ws-super-secret-value".to_string(),
        };
        let dbg = format!("{token:?}");
        assert!(!dbg.contains("wk-super-secret-id"), "leaked key: {dbg}");
        assert!(
            !dbg.contains("ws-super-secret-value"),
            "leaked secret: {dbg}"
        );
        assert!(
            dbg.contains("<redacted>"),
            "expected redaction marker: {dbg}"
        );
    }

    #[test]
    fn test_config_debug_redacts_all_secret_fields() {
        // `Config` is `{:?}`-formatted at startup (and in tests), so EVERY
        // credential-bearing field must be redacted: the client API bearers
        // (auth_tokens), the admin bearer (admin_token AND the config_service_token
        // derived from it), and both halves of the Modal proxy token. Non-secret
        // fields must stay visible for debuggability.
        with_env(
            &[
                ("SIE_AUTH_MODE", "token"),
                ("SIE_AUTH_TOKENS", "tok-secret-1,tok-secret-2"),
                ("SIE_ADMIN_TOKEN", "super-admin-secret"),
                ("SIE_MODAL_PROXY_TOKEN_ID", "wk-id-secret"),
                ("SIE_MODAL_PROXY_TOKEN_SECRET", "ws-value-secret"),
                ("SIE_NATS_URL", "nats://nats-host:4222"),
            ],
            || {
                let cfg = Config::load();
                // Sanity: the secrets really are loaded (so a green assert below
                // means redaction, not an empty field).
                assert_eq!(cfg.admin_token, "super-admin-secret");
                assert_eq!(
                    cfg.config_service_token.as_deref(),
                    Some("super-admin-secret")
                );
                assert_eq!(cfg.auth_tokens.len(), 2);

                let dbg = format!("{cfg:?}");
                for leaked in [
                    "super-admin-secret", // admin_token + config_service_token
                    "tok-secret-1",
                    "tok-secret-2", // auth_tokens
                    "wk-id-secret",
                    "ws-value-secret", // config_modal_proxy_token
                ] {
                    assert!(
                        !dbg.contains(leaked),
                        "Config Debug leaked {leaked:?}: {dbg}"
                    );
                }
                assert!(
                    dbg.contains("<redacted>"),
                    "expected redaction marker: {dbg}"
                );
                // Non-secret connection/config fields stay visible.
                assert!(
                    dbg.contains("nats-host"),
                    "non-secret nats_url must stay visible: {dbg}"
                );
            },
        );
    }

    // ── audit_auth ─────────────────────────────────────────────────

    fn cfg_with_auth(
        mode: &str,
        tokens: Vec<&str>,
        admin: &str,
        exempt_operational: bool,
    ) -> Config {
        let mut cfg = Config {
            host: String::new(),
            port: 0,
            worker_urls: Vec::new(),
            use_kubernetes: false,
            k8s_namespace: String::new(),
            k8s_service: String::new(),
            k8s_port: 0,
            health_mode: String::new(),
            nats_url: String::new(),
            nats_config_trusted_producers: Vec::new(),
            auth_mode: mode.to_string(),
            auth_tokens: tokens.into_iter().map(String::from).collect(),
            admin_token: admin.to_string(),
            auth_exempt_operational: exempt_operational,
            log_level: String::new(),
            json_logs: false,
            enable_pools: false,
            hot_reload: false,
            watch_polling: false,
            multi_router: false,
            request_timeout: 0.0,
            max_stream_pending: 0,
            stream_max_age_s: 0,
            configured_gpus: Vec::new(),
            gpu_profile_map: HashMap::new(),
            configured_physical_lanes: PhysicalLaneCatalog::default(),
            static_queue_pools: Vec::new(),
            model_aliases: HashMap::new(),
            bundles_dir: String::new(),
            models_dir: String::new(),
            config_service_url: None,
            config_service_token: None,
            config_modal_proxy_token: None,
            payload_store_url: String::new(),
            public_base_url: None,
        };
        // Silence the "unused mut" warning on the path where we don't mutate.
        let _ = &mut cfg;
        cfg
    }

    #[test]
    fn test_audit_auth_none_with_tokens_is_error() {
        let cfg = cfg_with_auth("none", vec!["t1"], "admin", false);
        let issues = cfg.audit_auth();
        assert!(
            issues
                .iter()
                .any(|(lvl, msg)| *lvl == AuditLevel::Error && msg.contains("DISABLED")),
            "expected error about tokens + disabled auth, got {:?}",
            issues
        );
    }

    #[test]
    fn test_audit_auth_token_mode_accepted() {
        let cfg = cfg_with_auth("token", vec!["t1"], "admin", false);
        let issues = cfg.audit_auth();
        assert!(
            issues.iter().all(|(lvl, _)| *lvl != AuditLevel::Error),
            "unexpected errors: {:?}",
            issues
        );
    }

    #[test]
    fn test_audit_auth_static_mode_accepted() {
        let cfg = cfg_with_auth("static", vec!["t1"], "admin", false);
        let issues = cfg.audit_auth();
        assert!(issues.iter().all(|(lvl, _)| *lvl != AuditLevel::Error));
    }

    #[test]
    fn test_audit_auth_enabled_without_tokens_is_error() {
        let cfg = cfg_with_auth("token", vec![], "", false);
        let issues = cfg.audit_auth();
        assert!(issues
            .iter()
            .any(|(lvl, msg)| *lvl == AuditLevel::Error && msg.contains("SIE_AUTH_TOKEN")));
    }

    #[test]
    fn test_audit_auth_enabled_without_admin_token_is_warn() {
        let cfg = cfg_with_auth("token", vec!["t1"], "", false);
        let issues = cfg.audit_auth();
        assert!(issues
            .iter()
            .any(|(lvl, msg)| *lvl == AuditLevel::Warn && msg.contains("SIE_ADMIN_TOKEN")));
    }

    #[test]
    fn test_audit_auth_unknown_mode_is_error() {
        let cfg = cfg_with_auth("bearer", vec!["t1"], "admin", false);
        let issues = cfg.audit_auth();
        assert!(issues
            .iter()
            .any(|(lvl, msg)| *lvl == AuditLevel::Error && msg.contains("not recognized")));
    }

    #[test]
    fn test_audit_auth_exempt_operational_is_warn() {
        let cfg = cfg_with_auth("token", vec!["t1"], "admin", true);
        let issues = cfg.audit_auth();
        assert!(issues
            .iter()
            .any(|(lvl, msg)| *lvl == AuditLevel::Warn
                && msg.contains("SIE_AUTH_EXEMPT_OPERATIONAL")));
    }

    #[test]
    fn test_audit_auth_clean_none_no_findings() {
        let cfg = cfg_with_auth("none", vec![], "", false);
        let issues = cfg.audit_auth();
        assert!(issues.is_empty(), "expected no findings, got {:?}", issues);
    }

    /// Guard: the gateway does not own a config store. Setting
    /// `SIE_CONFIG_STORE_DIR` or `SIE_CONFIG_RESTORE` must not resurrect a
    /// config-store path or mutate `Config`. If a future change reintroduces
    /// a field that reads either variable, this test has to be updated
    /// deliberately.
    #[test]
    fn test_removed_config_store_env_vars_are_ignored() {
        with_env(
            &[
                ("SIE_CONFIG_STORE_DIR", "/var/lib/gateway/config-store"),
                ("SIE_CONFIG_RESTORE", "true"),
            ],
            || {
                let cfg = Config::load();
                // No field on Config reads either var; this test guards against
                // a future accidental re-introduction being done via a field we
                // forgot to check. If someone adds config_store_dir back, this
                // test has to be updated deliberately.
                assert!(cfg.config_service_url.is_none());
                // The Debug impl for Config MUST NOT contain the removed paths.
                let dbg = format!("{:?}", cfg);
                assert!(
                    !dbg.contains("config_store_dir"),
                    "Config resurrected config_store_dir: {}",
                    dbg
                );
                assert!(
                    !dbg.contains("config_restore"),
                    "Config resurrected config_restore: {}",
                    dbg
                );
            },
        );
    }
}
