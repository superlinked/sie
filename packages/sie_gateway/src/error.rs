use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

#[allow(dead_code)]
#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("no healthy workers available")]
    NoWorkerAvailable,

    #[error("GPU provisioning in progress: {0}")]
    GpuProvisioning(String),

    #[error("worker connection error: {0}")]
    UpstreamConnection(String),

    #[error("internal error: {0}")]
    Internal(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            AppError::NoWorkerAvailable => (StatusCode::SERVICE_UNAVAILABLE, self.to_string()),
            AppError::GpuProvisioning(_) => (StatusCode::ACCEPTED, self.to_string()),
            AppError::UpstreamConnection(_) => (StatusCode::BAD_GATEWAY, self.to_string()),
            AppError::Internal(_) => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
        };

        let body = serde_json::json!({ "message": message });
        (status, axum::Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn response_status(err: AppError) -> StatusCode {
        err.into_response().status()
    }

    #[test]
    fn test_no_worker_available_is_503() {
        assert_eq!(
            response_status(AppError::NoWorkerAvailable),
            StatusCode::SERVICE_UNAVAILABLE
        );
    }

    #[test]
    fn test_gpu_provisioning_is_202() {
        assert_eq!(
            response_status(AppError::GpuProvisioning("l4".into())),
            StatusCode::ACCEPTED
        );
    }

    #[test]
    fn test_upstream_connection_is_502() {
        assert_eq!(
            response_status(AppError::UpstreamConnection("refused".into())),
            StatusCode::BAD_GATEWAY
        );
    }

    #[test]
    fn test_internal_is_500() {
        assert_eq!(
            response_status(AppError::Internal("oops".into())),
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }

    #[test]
    fn test_error_display() {
        assert_eq!(
            AppError::NoWorkerAvailable.to_string(),
            "no healthy workers available"
        );
        assert_eq!(
            AppError::GpuProvisioning("l4".into()).to_string(),
            "GPU provisioning in progress: l4"
        );
    }
}
