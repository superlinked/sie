//! Bounded lifecycle observations. These are sampled traces/logs, never metrics.
//! Body completion means the server consumed EOF, not client receipt of bytes.

use std::time::Instant;

use opentelemetry::trace::Status;
use tracing_opentelemetry::OpenTelemetrySpanExt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Success,
    Rejected,
    Error,
    Cancelled,
}

impl Outcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Rejected => "rejected",
            Self::Error => "error",
            Self::Cancelled => "cancelled",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorClass {
    None,
    ClientError,
    ServerError,
    Transport,
    Timeout,
    Worker,
    Cancelled,
}

impl ErrorClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::ClientError => "client_error",
            Self::ServerError => "server_error",
            Self::Transport => "transport",
            Self::Timeout => "timeout",
            Self::Worker => "worker",
            Self::Cancelled => "cancelled",
        }
    }
}

pub struct Lifecycle {
    span: tracing::Span,
    phase: &'static str,
    operation: &'static str,
    started: Instant,
    first_token_ms: Option<f64>,
    finished: bool,
}

impl Lifecycle {
    pub fn request(span: tracing::Span, operation: &'static str) -> Self {
        Self::from_span(span, "request", operation)
    }

    pub fn response_body(parent: opentelemetry::Context, operation: &'static str) -> Self {
        let span =
            tracing::info_span!("gateway.response_body", otel.name = "gateway.response_body");
        let _ = span.set_parent(parent);
        Self::from_span(span, "response_body", operation)
    }

    pub fn generation_stream(parent: opentelemetry::Context) -> Self {
        let span = tracing::info_span!(
            "gateway.generation_stream",
            otel.name = "gateway.generation_stream"
        );
        let _ = span.set_parent(parent);
        Self::from_span(span, "generation_stream", "generate")
    }

    fn from_span(span: tracing::Span, phase: &'static str, operation: &'static str) -> Self {
        Self {
            span,
            phase,
            operation,
            started: Instant::now(),
            first_token_ms: None,
            finished: false,
        }
    }

    pub fn context(&self) -> opentelemetry::Context {
        self.span.context()
    }

    /// First non-empty text/tool delta observed by the SSE driver. This is not
    /// adapter TTFT and creates no histogram or per-token observation.
    pub fn first_token(&mut self) {
        self.first_token_ms
            .get_or_insert_with(|| self.started.elapsed().as_secs_f64() * 1000.0);
    }

    pub fn finish_http(&mut self, status: u16) {
        let (outcome, class) = match status {
            500..=599 => (Outcome::Error, ErrorClass::ServerError),
            400..=499 => (Outcome::Rejected, ErrorClass::ClientError),
            _ => (Outcome::Success, ErrorClass::None),
        };
        self.finish(outcome, class);
    }

    pub fn finish(&mut self, outcome: Outcome, error_class: ErrorClass) {
        if self.finished {
            return;
        }
        self.finished = true;
        // No free-form status description: the structural code survives remote
        // privacy filtering. Client rejection/cancellation are not server errors.
        if outcome == Outcome::Error {
            self.span.set_status(Status::error(""));
        }
        super::tracing::record_lifecycle_log(
            &self.span.context(),
            self.phase,
            self.operation,
            outcome,
            error_class,
            self.started.elapsed().as_secs_f64() * 1000.0,
            self.first_token_ms,
        );
        // End at the terminal observation even if the body object is retained.
        self.span = tracing::Span::none();
    }
}

impl Drop for Lifecycle {
    fn drop(&mut self) {
        self.finish(Outcome::Cancelled, ErrorClass::Cancelled);
    }
}
