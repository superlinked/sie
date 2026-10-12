use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use crate::http_error::{code as err_code, json_detail};

#[allow(dead_code)]
#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("no healthy workers available")]
    NoWorkerAvailable,

    #[error("worker connection error: {0}")]
    UpstreamConnection(String),

    #[error("internal error: {0}")]
    Internal(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, code, message) = match &self {
            AppError::NoWorkerAvailable => (
                StatusCode::SERVICE_UNAVAILABLE,
                err_code::QUEUE_UNAVAILABLE,
                self.to_string(),
            ),
            AppError::UpstreamConnection(_) => (
                StatusCode::BAD_GATEWAY,
                err_code::INTERNAL_ERROR,
                self.to_string(),
            ),
            AppError::Internal(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                err_code::INTERNAL_ERROR,
                self.to_string(),
            ),
        };

        let body = json_detail(code, message);
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
    }
}
