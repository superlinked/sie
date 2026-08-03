//! Environment-variable helpers shared by every telemetry surface.

use std::env;

/// Read an env var, trimming surrounding whitespace and treating a
/// whitespace-only value as absent so it can't shadow a valid fallback.
pub fn cleaned_env(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Positive-truthy flag parse used by the `SIE_*_ENABLED` gates: exactly
/// `true` / `1` / `yes` (case-insensitive, trimmed) enable; everything else —
/// including absence — is off.
pub fn tracing_flag_set(raw: Option<&str>) -> bool {
    raw.is_some_and(|value| {
        let value = value.trim();
        value.eq_ignore_ascii_case("true") || value == "1" || value.eq_ignore_ascii_case("yes")
    })
}

/// `SIE_TRACING_ENABLED` gate for the trace exporter.
pub fn sie_tracing_enabled() -> bool {
    let raw = env::var("SIE_TRACING_ENABLED").ok();
    tracing_flag_set(raw.as_deref())
}

/// `SIE_METRICS_ENABLED` gate for the canonical metric exporter.
pub fn sie_metrics_enabled() -> bool {
    let raw = env::var("SIE_METRICS_ENABLED").ok();
    tracing_flag_set(raw.as_deref())
}

#[cfg(test)]
mod tests {
    use super::tracing_flag_set;

    #[test]
    fn tracing_flag_set_accepts_only_positive_truthy_values() {
        assert!(tracing_flag_set(Some("true")));
        assert!(tracing_flag_set(Some("1")));
        assert!(tracing_flag_set(Some("yes")));
        assert!(tracing_flag_set(Some("TRUE")));
        assert!(tracing_flag_set(Some(" true ")));

        assert!(!tracing_flag_set(Some("false")));
        assert!(!tracing_flag_set(Some("")));
        assert!(!tracing_flag_set(Some("   ")));
        assert!(!tracing_flag_set(None));
    }
}
