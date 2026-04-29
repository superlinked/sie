use std::collections::HashMap;
use std::env;

#[derive(Debug, Clone)]
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
    /// Opt-in bypass for operational surfaces (`/`, `/health`,
    /// `/metrics`, `/ws/*`) when auth is enabled. Kubernetes probes
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

    // Configured GPUs (survives scale-to-zero)
    pub configured_gpus: Vec<String>,
    // Pre-computed lowercase→original map for GPU profile resolution (avoids HashMap rebuild per request)
    pub gpu_profile_map: HashMap<String, String>,

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

    // Payload store (local path, s3://bucket/prefix, or gs://bucket/prefix)
    pub payload_store_url: String,
}

fn env_bool(key: &str) -> bool {
    match env::var(key) {
        Ok(v) => matches!(v.to_lowercase().as_str(), "true" | "1" | "yes"),
        Err(_) => false,
    }
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

            request_timeout: env_float("SIE_GATEWAY_REQUEST_TIMEOUT", 30.0),
            max_stream_pending: env_u64("SIE_GATEWAY_MAX_STREAM_PENDING", 50_000),

            configured_gpus: env_csv("SIE_GATEWAY_CONFIGURED_GPUS"),
            gpu_profile_map: {
                let gpus = env_csv("SIE_GATEWAY_CONFIGURED_GPUS");
                gpus.iter().map(|g| (g.to_lowercase(), g.clone())).collect()
            },

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

            payload_store_url: env_default("SIE_PAYLOAD_STORE_URL", "payload_store"),
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
                "SIE_AUTH_EXEMPT_OPERATIONAL=true: status page, /health, /metrics, and /ws/* bypass auth. Use only when those endpoints are already network-isolated.".to_string(),
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
        let _guard = ENV_LOCK.lock().unwrap();
        let old: Vec<(&str, Option<String>)> =
            vars.iter().map(|(k, _)| (*k, env::var(k).ok())).collect();
        for (k, v) in vars {
            env::set_var(k, v);
        }
        f();
        for (k, old_val) in old {
            match old_val {
                Some(v) => env::set_var(k, v),
                None => env::remove_var(k),
            }
        }
    }

    fn without_env<F: FnOnce()>(keys: &[&str], f: F) {
        let _guard = ENV_LOCK.lock().unwrap();
        let old: Vec<(&str, Option<String>)> =
            keys.iter().map(|k| (*k, env::var(k).ok())).collect();
        for k in keys {
            env::remove_var(k);
        }
        f();
        for (k, old_val) in old {
            if let Some(v) = old_val {
                env::set_var(k, v);
            }
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
            assert_eq!(cfg.payload_store_url, "payload_store");
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
            configured_gpus: Vec::new(),
            gpu_profile_map: HashMap::new(),
            bundles_dir: String::new(),
            models_dir: String::new(),
            config_service_url: None,
            config_service_token: None,
            payload_store_url: String::new(),
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
