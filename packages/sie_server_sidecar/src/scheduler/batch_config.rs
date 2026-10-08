//! [`BatchConfig`] — the shared-mutable caps the batcher consults on
//! every flush decision. Ported from `sie_server/core/batcher.py`.
//!
//! The Python implementation holds this as a plain mutable object and
//! relies on the GIL to serialise reads and writes from the controller
//! vs the batcher loop. In the Rust port we keep the shape identical
//! but hand the `AdaptiveBatchController` an `arc_swap::ArcSwap`
//! snapshot so a flush decision always sees a consistent set of knobs
//! even mid-update — see [`crate::scheduler::BatchFormer`] and
//! [`crate::scheduler::AdaptiveBatchController`] for the wiring.
//!
//! `ArcSwap` isn't pulled in as a dep for this first landing; the
//! controller → batcher channel is a simple `ArcSwap`-equivalent in
//! the follow-up wiring. For now the `BatchFormer` takes a
//! `BatchConfig` by value (with per-former update via
//! `update_config`), which matches Python's "mutate the field in
//! place" semantics closely enough for the primitive tests.
//!
//! Most default values match the Python defaults line-for-line; the
//! two intentional divergences (`coalesce_ms` and `max_batch_requests`)
//! are documented on [`BatchConfig::default`] and locked by the
//! `defaults_diverge_from_python_intentionally` test below.

/// Batch formation caps.
///
/// Ported from `sie_server.core.batcher.BatchConfig`.
///
/// Cost semantics are modality-agnostic:
/// * Text: `cost = token count`
/// * Images: `cost = 1` per image
/// * Mixed: component sum
///
/// Defaults intentionally mirror Python's defaults for parity testing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BatchConfig {
    /// Maximum total cost across all items in a batch before a flush.
    pub max_batch_cost: u64,

    /// Maximum number of items in a batch before a flush.
    pub max_batch_requests: usize,

    /// Maximum wall-clock wait since the *first* pending item arrived
    /// before a timeout flush.
    pub max_batch_wait_ms: f64,

    /// Coalesce window ceiling. See [`BatchConfig::effective_coalesce_ms`]
    /// for the effective value at runtime.
    ///
    /// Intent: set this to the upstream IPC inter-arrival jitter so
    /// a burst of 64 requests lands in a single GPU forward rather
    /// than being split across 2–3 under-filled ones.
    pub coalesce_ms: f64,

    /// Coalesce window as a fraction of [`Self::max_batch_wait_ms`].
    ///
    /// The effective window is
    /// `min(coalesce_ms, max_batch_wait_ms * coalesce_ratio)` — this
    /// keeps the coalesce time proportional when the adaptive
    /// controller mutates `max_batch_wait_ms`.
    pub coalesce_ratio: f64,
}

impl BatchConfig {
    /// Construct with Python-parity defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct with Python-parity defaults, optionally overridden
    /// by env vars. Used by the scheduler registry on startup.
    ///
    /// | Var                                 | Field                | Default | Notes |
    /// |-------------------------------------|----------------------|---------|-------|
    /// | `SIE_BATCHER_MAX_BATCH_WAIT_MS`     | `max_batch_wait_ms`  | 15.0    | Static flush-trigger ceiling — `AdaptiveBatchController` overwrites this on every step but the initial value matters during cold-start (see Python `core/batcher.py:BatchConfig`). |
    /// | `SIE_BATCHER_MAX_BATCH_COST`        | `max_batch_cost`     | 16384   | Static cost cap initial value. When set, it pins every model's cap and adaptive cost range; unset, each model uses its profile's `max_batch_tokens` (see [`crate::scheduler::SchedulerRegistry`]), else 16384. |
    /// | `SIE_BATCHER_MAX_BATCH_REQUESTS`    | `max_batch_requests` | **12**  | Hard count cap (controller does not move). **Diverges from Python's 64** — see [`BatchConfig::default`] for the runtime rationale. |
    /// | `SIE_BATCHER_COALESCE_MS`           | `coalesce_ms`        | **5.0** | Coalesce window ceiling. **Diverges from Python's 15.0** — see [`BatchConfig::default`] for the runtime rationale. |
    /// | `SIE_BATCHER_COALESCE_RATIO`        | `coalesce_ratio`     | 0.5     | Coalesce as fraction of `max_batch_wait_ms`. |
    ///
    /// Bad values (non-numeric, NaN) log a warning and fall back to
    /// the default — never crash the registry init.
    #[must_use]
    pub fn from_env_or_default() -> Self {
        let mut c = Self::default();

        fn get_f64(var: &str) -> Option<f64> {
            std::env::var(var).ok().and_then(|s| {
                s.parse::<f64>().ok().filter(|v| v.is_finite()).or_else(|| {
                    tracing::warn!(
                        var = var,
                        raw = s.as_str(),
                        "batcher env var: ignoring non-numeric/non-finite value"
                    );
                    None
                })
            })
        }
        fn get_usize(var: &str) -> Option<usize> {
            std::env::var(var)
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
        }

        if let Some(v) = get_f64("SIE_BATCHER_MAX_BATCH_WAIT_MS") {
            c.max_batch_wait_ms = v;
        }
        if let Some(v) = Self::max_batch_cost_env_override() {
            c.max_batch_cost = v;
        }
        if let Some(v) = get_usize("SIE_BATCHER_MAX_BATCH_REQUESTS") {
            c.max_batch_requests = v;
        }
        if let Some(v) = get_f64("SIE_BATCHER_COALESCE_MS") {
            c.coalesce_ms = v;
        }
        if let Some(v) = get_f64("SIE_BATCHER_COALESCE_RATIO") {
            c.coalesce_ratio = v;
        }

        tracing::info!(
            max_batch_cost = c.max_batch_cost,
            max_batch_requests = c.max_batch_requests,
            max_batch_wait_ms = c.max_batch_wait_ms,
            coalesce_ms = c.coalesce_ms,
            coalesce_ratio = c.coalesce_ratio,
            "batcher: BatchConfig constructed from env+defaults"
        );

        c
    }

    /// The operator's static cost cap, `SIE_BATCHER_MAX_BATCH_COST`, when it
    /// is set to an unsigned integer.
    ///
    /// When present it wins over each model's own `max_batch_tokens`: the
    /// scheduler registry is then built cost-pinned and every model keeps
    /// this cap and the adaptive range derived from it, as before per-model
    /// budgets existed. Unparseable values are ignored, as in
    /// [`Self::from_env_or_default`].
    #[must_use]
    pub fn max_batch_cost_env_override() -> Option<u64> {
        parse_max_batch_cost(std::env::var("SIE_BATCHER_MAX_BATCH_COST").ok().as_deref())
    }

    /// Effective coalesce window used by the batcher.
    ///
    /// Mirrors Python's
    /// `min(coalesce_ms, max_batch_wait_ms * coalesce_ratio)` exactly
    /// — f64 math on both sides means identical outputs bit-for-bit
    /// within IEEE 754 determinism.
    #[must_use]
    pub fn effective_coalesce_ms(&self) -> f64 {
        f64::min(
            self.coalesce_ms,
            self.max_batch_wait_ms * self.coalesce_ratio,
        )
    }

    /// Python-parity alias for [`Self::max_batch_cost`] mirroring the
    /// Python `max_batch_tokens` property.
    #[must_use]
    pub fn max_batch_tokens(&self) -> u64 {
        self.max_batch_cost
    }
}

fn parse_max_batch_cost(raw: Option<&str>) -> Option<u64> {
    raw.and_then(|s| s.parse::<u64>().ok())
}

impl Default for BatchConfig {
    /// Defaults: most fields mirror Python's `BatchConfig` line-for-line.
    ///
    /// **Two intentional divergences** from Python:
    ///
    /// | Field                | Python default | Rust default | Why diverge |
    /// |----------------------|----------------|--------------|-------------|
    /// | `coalesce_ms`        | 15.0           | **5.0**      | The tight `FetchExpiryController` already flushes the upstream pull-loop on a 2-15 ms quantum, so another 15 ms downstream wait is latency padding. 5 ms keeps a small coalesce window without crossing into under-batching. |
    /// | `max_batch_requests` | 64             | **12**       | The tight pull quantum naturally produces small fetch batches. 12 is comfortable headroom for arrival bursts without allowing pathological single-batch growth. |
    ///
    /// The divergences are locked by
    /// `tests::defaults_diverge_from_python_intentionally` so Python parity
    /// is not restored by accident; retunes should update this rationale and
    /// the test together.
    fn default() -> Self {
        Self {
            max_batch_cost: 16_384,
            max_batch_requests: 12,
            max_batch_wait_ms: 15.0,
            coalesce_ms: 5.0,
            coalesce_ratio: 0.5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_python_for_non_diverged_fields() {
        let c = BatchConfig::default();
        // Python-parity fields (must NOT silently drift):
        assert_eq!(c.max_batch_cost, 16_384);
        assert!((c.max_batch_wait_ms - 15.0).abs() < f64::EPSILON);
        assert!((c.coalesce_ratio - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn defaults_diverge_from_python_intentionally() {
        // Locks the two intentional divergences from Python defaults.
        // Retunes should update `BatchConfig::default` and this test
        // together so the runtime rationale stays attached to the guard.
        let c = BatchConfig::default();
        // Python's `BatchConfig.coalesce_ms = 15.0`; we ship 5.0.
        assert!(
            (c.coalesce_ms - 5.0).abs() < f64::EPSILON,
            "expected coalesce_ms=5.0 (Rust-side worker default), got {}",
            c.coalesce_ms,
        );
        // Python's `BatchConfig.max_batch_requests = 64`; we ship 12.
        assert_eq!(
            c.max_batch_requests, 12,
            "expected max_batch_requests=12 (Rust-side worker default)"
        );
    }

    #[test]
    fn effective_coalesce_is_min_of_ceiling_and_proportional() {
        let c = BatchConfig {
            coalesce_ms: 15.0,
            max_batch_wait_ms: 10.0, // proportional = 5.0
            coalesce_ratio: 0.5,
            ..BatchConfig::default()
        };
        // proportional wins because 5.0 < 15.0
        assert!((c.effective_coalesce_ms() - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn effective_coalesce_ceiling_clamps_upward() {
        let c = BatchConfig {
            coalesce_ms: 2.0,
            max_batch_wait_ms: 100.0, // proportional = 50.0
            coalesce_ratio: 0.5,
            ..BatchConfig::default()
        };
        // ceiling wins because 2.0 < 50.0
        assert!((c.effective_coalesce_ms() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn max_batch_cost_override_parses_only_unsigned_integers() {
        // Same acceptance as the `max_batch_cost` field of
        // `from_env_or_default`: anything else leaves the cap unpinned.
        assert_eq!(parse_max_batch_cost(Some("720000")), Some(720_000));
        assert_eq!(parse_max_batch_cost(Some("16384")), Some(16_384));
        assert_eq!(parse_max_batch_cost(None), None);
        assert_eq!(parse_max_batch_cost(Some("")), None);
        assert_eq!(parse_max_batch_cost(Some("-1")), None);
        assert_eq!(parse_max_batch_cost(Some("16k")), None);
    }

    #[test]
    fn max_batch_tokens_alias() {
        let c = BatchConfig {
            max_batch_cost: 4242,
            ..BatchConfig::default()
        };
        assert_eq!(c.max_batch_tokens(), 4242);
    }
}
