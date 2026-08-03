//! Modal proxy authentication for OTLP/HTTP export to the managed collector.
//!
//! The managed data plane fronts its collector with a Modal proxy that
//! requires a `Modal-Key`/`Modal-Secret` credential pair. Headers are
//! attached only after the exact-origin trust check succeeds, and only over
//! HTTPS to a `.modal.run` origin.

use std::collections::HashMap;
use std::time::Duration;

use crate::env::cleaned_env;
use crate::transport::OtlpProtocol;

/// True only for the explicit managed Modal proxy-auth posture.
pub fn modal_proxy_auth_enabled() -> bool {
    cleaned_env("SIE_MODAL_PROXY_AUTH").is_some_and(|value| {
        matches!(
            value.to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

/// Canonicalize a trusted Modal HTTPS URL, or reject it.
pub fn trusted_modal_origin(
    raw: &str,
    origin_only: bool,
    expected_path: Option<&str>,
) -> Option<String> {
    let parsed = reqwest::Url::parse(raw).ok()?;
    let host = parsed.host_str()?.to_ascii_lowercase();
    if parsed.scheme() != "https"
        || !host.ends_with(".modal.run")
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.port_or_known_default() != Some(443)
        || parsed.query().is_some()
        || parsed.fragment().is_some()
        || (origin_only && parsed.path() != "/")
        || expected_path.is_some_and(|path| parsed.path() != path)
    {
        return None;
    }
    Some(format!("https://{host}"))
}

/// Return Modal headers only for the exact provisioner-resolved collector,
/// reading the credential pair + allowed origin from the environment.
pub fn modal_proxy_headers(
    endpoint: &str,
    expected_path: &str,
) -> Result<HashMap<String, String>, String> {
    modal_proxy_headers_from_values(
        endpoint,
        expected_path,
        modal_proxy_auth_enabled(),
        cleaned_env("SIE_OTEL_PROXY_AUTH_ORIGIN").as_deref(),
        cleaned_env("SIE_MODAL_PROXY_TOKEN_ID").as_deref(),
        cleaned_env("SIE_MODAL_PROXY_TOKEN_SECRET").as_deref(),
    )
}

pub fn modal_proxy_headers_from_values(
    endpoint: &str,
    expected_path: &str,
    proxy_auth_enabled: bool,
    allowed_origin: Option<&str>,
    token_id: Option<&str>,
    token_secret: Option<&str>,
) -> Result<HashMap<String, String>, String> {
    if !proxy_auth_enabled {
        return Ok(HashMap::new());
    }
    let allowed = allowed_origin
        .and_then(|origin| trusted_modal_origin(origin, true, None))
        .ok_or_else(|| "managed OTLP proxy-auth origin is missing or untrusted".to_string())?;
    let actual = trusted_modal_origin(endpoint, false, Some(expected_path))
        .ok_or_else(|| "managed OTLP signal endpoint is untrusted".to_string())?;
    if actual != allowed {
        return Err(
            "managed OTLP endpoint does not match the provisioned proxy-auth origin".to_string(),
        );
    }

    let (id, secret) = token_id
        .filter(|value| !value.trim().is_empty())
        .zip(token_secret.filter(|value| !value.trim().is_empty()))
        .ok_or_else(|| {
            "managed OTLP proxy authentication requires a complete Modal credential pair"
                .to_string()
        })?;
    Ok(HashMap::from([
        ("Modal-Key".to_string(), id.to_string()),
        ("Modal-Secret".to_string(), secret.to_string()),
    ]))
}

/// Blocking client for authenticated export. Redirects are disabled because
/// reqwest otherwise preserves custom Modal headers across redirects.
///
/// OTel 0.32's batch span processor and periodic metric reader poll exporter
/// futures on dedicated std threads without a Tokio reactor; a reqwest async
/// client panics there. The blocking client blocks only those SDK-owned
/// exporter threads, and constructing it on a short-lived helper thread also
/// remains safe when initialization itself runs inside a Tokio runtime.
pub fn authenticated_http_client() -> Result<reqwest::blocking::Client, String> {
    std::thread::spawn(|| {
        reqwest::blocking::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_secs(10))
            .build()
    })
    .join()
    .map_err(|_| "build no-redirect OTLP HTTP client panicked".to_string())?
    .map_err(|error| format!("build no-redirect OTLP HTTP client: {error}"))
}

/// Modal proxy authentication requires the HTTP transport; fail closed on the
/// gRPC path instead of silently exporting unauthenticated.
pub fn validate_modal_proxy_transport(
    protocol: OtlpProtocol,
    proxy_auth_enabled: bool,
) -> Result<(), String> {
    if proxy_auth_enabled && protocol != OtlpProtocol::Http {
        return Err("Modal OTLP proxy authentication requires HTTP transport".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modal_headers_require_exact_https_origin_path_and_complete_pair() {
        let headers = modal_proxy_headers_from_values(
            "https://workspace--collector.modal.run/v1/metrics",
            "/v1/metrics",
            true,
            Some("https://workspace--collector.modal.run"),
            Some("id"),
            Some("secret"),
        )
        .expect("trusted Modal signal endpoint");
        assert_eq!(headers.get("Modal-Key").map(String::as_str), Some("id"));
        assert_eq!(
            headers.get("Modal-Secret").map(String::as_str),
            Some("secret")
        );

        for endpoint in [
            "http://workspace--collector.modal.run/v1/metrics",
            "https://workspace--collector.modal.run.evil.example/v1/metrics",
            "https://other--collector.modal.run/v1/metrics",
            "https://workspace--collector.modal.run/v1/traces",
        ] {
            assert!(modal_proxy_headers_from_values(
                endpoint,
                "/v1/metrics",
                true,
                Some("https://workspace--collector.modal.run"),
                Some("id"),
                Some("secret"),
            )
            .is_err());
        }
        assert!(modal_proxy_headers_from_values(
            "https://workspace--collector.modal.run/v1/metrics",
            "/v1/metrics",
            true,
            Some("https://workspace--collector.modal.run"),
            Some("id"),
            None,
        )
        .is_err());
    }

    #[test]
    fn proxy_auth_disabled_yields_no_headers() {
        assert_eq!(
            modal_proxy_headers_from_values(
                "http://anything",
                "/v1/traces",
                false,
                None,
                None,
                None
            ),
            Ok(HashMap::new())
        );
    }

    #[tokio::test]
    async fn modal_proxy_auth_rejects_grpc_and_builds_dedicated_thread_client_inside_runtime() {
        assert!(validate_modal_proxy_transport(OtlpProtocol::Grpc, true).is_err());
        assert!(validate_modal_proxy_transport(OtlpProtocol::Http, true).is_ok());
        authenticated_http_client().expect("bounded no-redirect blocking client");
    }
}
