//! Native Rust backend dispatch for the worker IPC server.

use std::time::Instant;

use anyhow::Result;

use crate::candle_backend::CandleBackend;
use crate::ipc_types::{
    ApplyModelConfigRequest, BatchOutcome, Disposition, EnsureModelReadyResponse,
    ProcessEncodeBatchRequest, ProcessScoreBatchRequest, ReplaceModelConfigsRequest,
    RunBatchRequest, SetPinnedModelsRequest, SetPinnedModelsResponse,
};
use crate::observability::metrics::{
    self, AuthoritativeUnits, ItemCompleted, PhaseDurations, WorkerOutcome,
};

#[derive(Clone)]
pub enum NativeBackend {
    Candle(CandleBackend),
}

impl NativeBackend {
    pub fn supported_models(&self) -> Vec<String> {
        match self {
            Self::Candle(backend) => backend.supported_models(),
        }
    }

    pub fn loaded_models(&self) -> Vec<String> {
        match self {
            Self::Candle(backend) => backend.loaded_models(),
        }
    }

    pub async fn health_ready(&self) -> bool {
        match self {
            Self::Candle(backend) => backend.health_ready().await,
        }
    }

    pub fn start_idle_evictor(&self) -> bool {
        match self {
            Self::Candle(backend) => backend.start_idle_evictor(),
        }
    }

    pub fn stop_idle_evictor(&self) -> bool {
        match self {
            Self::Candle(backend) => backend.stop_idle_evictor(),
        }
    }

    pub async fn ensure_model_ready(&self, model_id: &str) -> EnsureModelReadyResponse {
        match self {
            Self::Candle(backend) => backend.ensure_model_ready(model_id).await,
        }
    }

    pub async fn process_encode_batch(&self, req: ProcessEncodeBatchRequest) -> BatchOutcome {
        match self {
            Self::Candle(backend) => {
                let telemetry = BatchTelemetry::for_encode(backend, &req);
                let outcome = backend.process_encode_batch(req).await;
                if let Some(telemetry) = telemetry {
                    telemetry.record(&outcome);
                }
                outcome
            }
        }
    }

    pub async fn process_score_batch(&self, req: ProcessScoreBatchRequest) -> BatchOutcome {
        match self {
            Self::Candle(backend) => {
                let telemetry = BatchTelemetry::for_score(backend, &req);
                let outcome = backend.process_score_batch(req).await;
                if let Some(telemetry) = telemetry {
                    telemetry.record(&outcome);
                }
                outcome
            }
        }
    }

    pub async fn run_batch(&self, req: RunBatchRequest) -> BatchOutcome {
        match self {
            Self::Candle(backend) => {
                let telemetry = BatchTelemetry::for_run_batch(backend, &req);
                let outcome = backend.run_batch(req).await;
                if let Some(telemetry) = telemetry {
                    telemetry.record(&outcome);
                }
                outcome
            }
        }
    }

    pub fn apply_model_config(&self, req: &ApplyModelConfigRequest) -> Result<()> {
        match self {
            Self::Candle(backend) => backend.apply_model_config(req),
        }
    }

    pub fn replace_model_configs(&self, req: &ReplaceModelConfigsRequest) -> Result<Vec<String>> {
        match self {
            Self::Candle(backend) => backend.replace_model_configs(req),
        }
    }

    pub fn set_preload_models(&self, models: &[String]) -> u32 {
        match self {
            Self::Candle(backend) => backend.set_preload_models(models),
        }
    }

    pub fn set_pinned_models(&self, req: &SetPinnedModelsRequest) -> SetPinnedModelsResponse {
        match self {
            Self::Candle(backend) => backend.set_pinned_models(req),
        }
    }
}

struct ItemTelemetryDimensions {
    operation: &'static str,
    model: String,
    profile: String,
}

struct BatchTelemetry {
    started: Instant,
    items: Vec<ItemTelemetryDimensions>,
}

impl BatchTelemetry {
    fn for_encode(backend: &CandleBackend, req: &ProcessEncodeBatchRequest) -> Option<Self> {
        if !metrics::metrics_enabled() {
            return None;
        }
        Some(Self {
            started: Instant::now(),
            items: req
                .items
                .iter()
                .map(|item| {
                    let (model, profile) =
                        backend.telemetry_dimensions(&req.model_id, item.profile_id.as_deref());
                    ItemTelemetryDimensions {
                        operation: "encode",
                        model,
                        profile,
                    }
                })
                .collect(),
        })
    }

    fn for_score(backend: &CandleBackend, req: &ProcessScoreBatchRequest) -> Option<Self> {
        if !metrics::metrics_enabled() {
            return None;
        }
        Some(Self {
            started: Instant::now(),
            items: req
                .items
                .iter()
                .map(|item| {
                    let (model, profile) =
                        backend.telemetry_dimensions(&req.model_id, item.profile_id.as_deref());
                    ItemTelemetryDimensions {
                        operation: "score",
                        model,
                        profile,
                    }
                })
                .collect(),
        })
    }

    fn for_run_batch(backend: &CandleBackend, req: &RunBatchRequest) -> Option<Self> {
        if !metrics::metrics_enabled() {
            return None;
        }
        Some(Self {
            started: Instant::now(),
            items: req
                .items
                .iter()
                .map(|item| {
                    let requested_profile = item
                        .encode
                        .as_ref()
                        .and_then(|encode| encode.profile_id.as_deref())
                        .or_else(|| {
                            item.score
                                .as_ref()
                                .and_then(|score| score.profile_id.as_deref())
                        });
                    let (model, profile) =
                        backend.telemetry_dimensions(&req.model_id, requested_profile);
                    ItemTelemetryDimensions {
                        operation: bounded_operation(&item.op),
                        model,
                        profile,
                    }
                })
                .collect(),
        })
    }

    fn record(self, batch: &BatchOutcome) {
        let duration_s = self.started.elapsed().as_secs_f64();
        for (dimensions, outcome) in self.items.iter().zip(&batch.outcomes) {
            metrics::record_item_completed(ItemCompleted {
                operation: dimensions.operation,
                outcome: worker_outcome(&outcome.disposition),
                model: &dimensions.model,
                profile: &dimensions.profile,
                duration_s,
                phases: PhaseDurations {
                    tokenization_s: milliseconds_to_seconds(outcome.tokenization_ms),
                    inference_s: milliseconds_to_seconds(outcome.inference_ms),
                    postprocessing_s: milliseconds_to_seconds(outcome.postprocessing_ms),
                },
                // Rust/Candle does not yet return authoritative per-item input
                // units for every tokenization path. Never estimate from text.
                units: AuthoritativeUnits::default(),
            });
        }
    }
}

fn bounded_operation(operation: &str) -> &'static str {
    match operation {
        "encode" => "encode",
        "score" => "score",
        "extract" => "extract",
        _ => "other",
    }
}

fn worker_outcome(disposition: &Disposition) -> WorkerOutcome {
    match disposition {
        Disposition::PublishAndAck => WorkerOutcome::Success,
        Disposition::PublishErrorAndAck => WorkerOutcome::Error,
        Disposition::NakRetry => WorkerOutcome::Retry,
    }
}

fn milliseconds_to_seconds(value: Option<f64>) -> Option<f64> {
    value.map(|milliseconds| milliseconds / 1_000.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_dispositions_map_to_the_bounded_worker_outcomes() {
        assert_eq!(
            worker_outcome(&Disposition::PublishAndAck),
            WorkerOutcome::Success
        );
        assert_eq!(
            worker_outcome(&Disposition::PublishErrorAndAck),
            WorkerOutcome::Error
        );
        assert_eq!(worker_outcome(&Disposition::NakRetry), WorkerOutcome::Retry);
    }

    #[test]
    fn phase_durations_convert_milliseconds_to_contract_seconds() {
        assert_eq!(milliseconds_to_seconds(Some(250.0)), Some(0.25));
        assert_eq!(milliseconds_to_seconds(None), None);
    }
}
