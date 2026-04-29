use axum::body::Body;
use axum::http::Request;
use axum::response::Response;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Instant;
use tower::{Layer, Service};
use tracing::info;

#[derive(Clone)]
pub struct AuditLayer;

impl AuditLayer {
    pub fn new() -> Self {
        Self
    }
}

impl<S> Layer<S> for AuditLayer {
    type Service = AuditMiddleware<S>;

    fn layer(&self, inner: S) -> Self::Service {
        AuditMiddleware { inner }
    }
}

#[derive(Clone)]
pub struct AuditMiddleware<S> {
    inner: S,
}

impl<S> Service<Request<Body>> for AuditMiddleware<S>
where
    S: Service<Request<Body>, Response = Response> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = Response;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<Body>) -> Self::Future {
        let start = Instant::now();
        let method = req.method().to_string();
        let path = req.uri().path().to_string();

        let token_id = req
            .headers()
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .map(|h| {
                let token = if h.to_lowercase().starts_with("bearer ") {
                    h[7..].trim()
                } else {
                    h.trim()
                };
                mask_token(token)
            })
            .unwrap_or_default();

        let content_length = req
            .headers()
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(0);

        // Extract routing hints from headers
        let model = req
            .headers()
            .get("x-sie-model")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();

        let pool = req
            .headers()
            .get("x-sie-pool")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();

        let gpu = req
            .headers()
            .get("x-sie-machine-profile")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();

        let mut inner = self.inner.clone();

        Box::pin(async move {
            let response = inner.call(req).await?;

            let elapsed = start.elapsed();
            let status = response.status().as_u16();

            let worker = response
                .headers()
                .get("x-sie-worker")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("")
                .to_string();

            // Only audit non-health/metrics endpoints to reduce noise
            if !is_infrastructure_path(&path) {
                info!(
                    event = "api_request",
                    method = %method,
                    endpoint = %path,
                    status = status,
                    token_id = %token_id,
                    model = %model,
                    pool = %pool,
                    gpu = %gpu,
                    worker = %worker,
                    latency_ms = elapsed.as_millis() as u64,
                    body_bytes = content_length,
                    "audit"
                );
            }

            Ok(response)
        })
    }
}

fn is_infrastructure_path(path: &str) -> bool {
    matches!(path, "/health" | "/healthz" | "/readyz" | "/metrics")
}

fn mask_token(token: &str) -> String {
    if token.len() <= 4 {
        "****".to_string()
    } else {
        format!(
            "{}{}",
            "*".repeat(token.len() - 4),
            &token[token.len() - 4..]
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mask_token_short() {
        assert_eq!(mask_token("abc"), "****");
        assert_eq!(mask_token(""), "****");
    }

    #[test]
    fn test_mask_token_long() {
        assert_eq!(mask_token("secret-token-123"), "************-123");
    }

    #[test]
    fn test_mask_token_exact_4() {
        assert_eq!(mask_token("abcd"), "****");
    }

    #[test]
    fn test_mask_token_5_chars() {
        assert_eq!(mask_token("abcde"), "*bcde");
    }

    #[test]
    fn test_infrastructure_paths() {
        assert!(is_infrastructure_path("/health"));
        assert!(is_infrastructure_path("/healthz"));
        assert!(is_infrastructure_path("/readyz"));
        assert!(is_infrastructure_path("/metrics"));
        assert!(!is_infrastructure_path("/v1/encode/model"));
    }
}
