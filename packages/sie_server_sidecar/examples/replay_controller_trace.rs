//! Replay an adaptive-controller trace through the Rust
//! [`AdaptiveBatchController`] and emit one `step` output per line on
//! stdout.
//!
//! Used to validate that the Rust controller produces floating-point outputs
//! that are identical ULP-for-ULP to a reference implementation on captured
//! traces.
//!
//! ## Trace format (JSONL on stdin)
//!
//! Each line is one event of shape:
//!
//! ```json
//! {"event": "step", "observed_p50_ms": 45.2, "fill_ratio": 0.81, "batch_size": 32}
//! {"event": "record_inference", "inference_ms": 12.5}
//! ```
//!
//! * `step` events advance the controller; the response line carries
//!   the returned `(wait_ms, batch_cost)` plus `starvation_resets`.
//! * `record_inference` feeds the auto-calibration tracker. Emits no
//!   response line — skipped on the consumer side as well.
//!
//! Unknown fields are ignored (`serde(default)`), unknown events are
//! an error (fail-loud so a typo in the capture doesn't silently
//! skip events).
//!
//! ## CLI
//!
//! `--config <path>`: optional JSON file with controller construction
//! parameters (see `BuilderConfig` below). Defaults to the same
//! Python-parity defaults used in
//! `sie_server.config.engine.AdaptiveBatchingConfig`.
//!
//! Output is one JSONL line per *step* event (not per input line).

use std::io::{self, BufRead, BufWriter, Write};

use serde::{Deserialize, Serialize};
use sie_server_sidecar::scheduler::{AdaptiveBatchController, AdaptiveBatchControllerBuilder};

#[derive(Debug, Default, Deserialize)]
struct BuilderConfig {
    // Every field mirrors
    // `sie_server.config.engine.AdaptiveBatchingConfig`. Using double
    // `Option` for `target_p50_ms` so the JSON `null` explicitly opts
    // into auto-calibration; omitting the key falls back to the
    // builder default (None => auto-calibrate as well, so the two are
    // semantically equivalent — the wrapper exists for trace fidelity
    // when rolling out from a Python config that explicitly set
    // `target_p50_ms: null`).
    target_p50_ms: Option<Option<f64>>,
    calibration_multiplier: Option<f64>,
    min_target_p50_ms: Option<f64>,
    max_target_p50_ms: Option<f64>,
    min_wait_ms: Option<f64>,
    max_wait_ms: Option<f64>,
    min_batch_cost: Option<u64>,
    max_batch_cost: Option<u64>,
    gain: Option<f64>,
    integral_gain: Option<f64>,
    cost_gain: Option<f64>,
    update_interval: Option<u32>,
    fill_ratio_threshold: Option<f64>,
    starvation_recovery_enabled: Option<bool>,
    starvation_window: Option<u32>,
    starvation_batch_size: Option<u32>,
    initial_wait_ms: Option<f64>,
    initial_batch_cost: Option<u64>,
}

impl BuilderConfig {
    fn into_builder(self) -> AdaptiveBatchControllerBuilder {
        let mut b = AdaptiveBatchController::builder();
        if let Some(v) = self.target_p50_ms {
            b = b.target_p50_ms(v);
        }
        if let Some(v) = self.calibration_multiplier {
            b = b.calibration_multiplier(v);
        }
        if let Some(v) = self.min_target_p50_ms {
            b = b.min_target_p50_ms(v);
        }
        if let Some(v) = self.max_target_p50_ms {
            b = b.max_target_p50_ms(v);
        }
        if let Some(v) = self.min_wait_ms {
            b = b.min_wait_ms(v);
        }
        if let Some(v) = self.max_wait_ms {
            b = b.max_wait_ms(v);
        }
        if let Some(v) = self.min_batch_cost {
            b = b.min_batch_cost(v);
        }
        if let Some(v) = self.max_batch_cost {
            b = b.max_batch_cost(v);
        }
        if let Some(v) = self.gain {
            b = b.gain(v);
        }
        if let Some(v) = self.integral_gain {
            b = b.integral_gain(v);
        }
        if let Some(v) = self.cost_gain {
            b = b.cost_gain(v);
        }
        if let Some(v) = self.update_interval {
            b = b.update_interval(v);
        }
        if let Some(v) = self.fill_ratio_threshold {
            b = b.fill_ratio_threshold(v);
        }
        if let Some(v) = self.starvation_recovery_enabled {
            b = b.starvation_recovery_enabled(v);
        }
        if let Some(v) = self.starvation_window {
            b = b.starvation_window(v);
        }
        if let Some(v) = self.starvation_batch_size {
            b = b.starvation_batch_size(v);
        }
        if let Some(v) = self.initial_wait_ms {
            b = b.initial_wait_ms(v);
        }
        if let Some(v) = self.initial_batch_cost {
            b = b.initial_batch_cost(v);
        }
        b
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
enum TraceEvent {
    Step {
        /// Elapsed seconds since the previous `step` event. Required
        /// (not `Option<f64>`) so a trace that forgets to capture
        /// `dt_s` fails loudly at parse time instead of silently
        /// driving a zero integrator on one side.
        dt_s: f64,
        #[serde(default)]
        observed_p50_ms: Option<f64>,
        #[serde(default)]
        fill_ratio: Option<f64>,
        #[serde(default)]
        batch_size: Option<usize>,
    },
    RecordInference {
        inference_ms: f64,
    },
}

#[derive(Debug, Serialize)]
struct StepOutput {
    wait_ms: f64,
    batch_cost: u64,
    /// Cumulative count — lets the diff tool verify starvation events
    /// landed on the same step index in both implementations, not just
    /// "the totals agree at the end".
    starvation_resets: u32,
}

fn main() -> anyhow::Result<()> {
    // Minimal CLI: `--config <path>`; everything else is unrecognised
    // on purpose so trace format stays the public interface.
    let mut args = std::env::args().skip(1);
    let mut config: BuilderConfig = BuilderConfig::default();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--config" => {
                let path = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--config requires a path argument"))?;
                let raw = std::fs::read_to_string(&path)?;
                config = serde_json::from_str(&raw)?;
            }
            "-h" | "--help" => {
                eprintln!(
                    "replay_controller_trace [--config CONFIG.json]\n\
                     Reads JSONL trace events from stdin, emits one JSONL\n\
                     step-output line per `step` event on stdout."
                );
                return Ok(());
            }
            other => anyhow::bail!("unknown argument: {other}"),
        }
    }

    let mut ctrl = config.into_builder().build();

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    for (lineno, line) in stdin.lock().lines().enumerate() {
        let raw = line?;
        let trimmed = raw.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let event: TraceEvent = serde_json::from_str(trimmed).map_err(|e| {
            anyhow::anyhow!("line {}: invalid JSON: {} ({})", lineno + 1, e, trimmed)
        })?;
        match event {
            TraceEvent::Step {
                dt_s,
                observed_p50_ms,
                fill_ratio,
                batch_size,
            } => {
                let (wait_ms, batch_cost) =
                    ctrl.step_replay(dt_s, observed_p50_ms, fill_ratio, batch_size);
                let snap = ctrl.snapshot(observed_p50_ms, fill_ratio);
                let output = StepOutput {
                    wait_ms,
                    batch_cost,
                    starvation_resets: snap.starvation_resets,
                };
                serde_json::to_writer(&mut out, &output)?;
                out.write_all(b"\n")?;
            }
            TraceEvent::RecordInference { inference_ms } => {
                ctrl.record_inference_sample(inference_ms);
            }
        }
    }

    out.flush()?;
    Ok(())
}
