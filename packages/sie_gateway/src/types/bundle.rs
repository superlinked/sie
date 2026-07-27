use serde::{Deserialize, Serialize};

/// Recognised execution engines for the ``engine`` bundle field.
///
/// `pytorch` is the Python ``sie_server`` adapter image. `candle` is
/// the Rust ``sie-server-rust`` worker process backed by native Candle
/// execution.
///
/// Mirrors ``sie_config.model_registry.KNOWN_ENGINES`` on the Python
/// side. Drift is a quiet mis-routing footgun, so both ends are kept
/// in lock-step.
pub const KNOWN_ENGINES: &[&str] = &["pytorch", "candle"];

/// Default engine when a bundle YAML omits the field. Same as the
/// Python worker path. Kept as a separate constant so the asymmetric
/// "default vs known" contract reads cleanly as more native engines
/// land.
pub const DEFAULT_ENGINE: &str = "pytorch";

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct BundleConfig {
    pub name: String,
    #[serde(default = "default_priority")]
    pub priority: i32,
    #[serde(default)]
    pub adapters: Vec<String>,
    #[serde(default)]
    pub default: bool,
    /// Execution engine the bundle's worker image speaks. Today
    /// ``"pytorch"`` and ``"candle"`` are recognised;
    /// see ``KNOWN_ENGINES`` for the load-time check. Defaults to
    /// ``"pytorch"`` for back-compat with pre-engine bundle YAMLs.
    #[serde(default = "default_engine")]
    pub engine: String,
    #[serde(default)]
    pub machine_profiles: Vec<BundleMachineProfile>,
    #[serde(default)]
    pub adapter_module: Option<String>,
    #[serde(skip)]
    pub config_hash: String,
}

#[allow(dead_code)]
fn default_priority() -> i32 {
    100
}

#[allow(dead_code)]
fn default_engine() -> String {
    DEFAULT_ENGINE.to_string()
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleMachineProfile {
    pub name: String,
    #[serde(default)]
    pub gpu_type: String,
    #[serde(default)]
    pub gpu_count: u32,
    #[serde(default)]
    pub max_batch_size: u32,
    #[serde(default)]
    pub max_sequence_length: u32,
}

#[derive(Debug, Clone)]
pub struct BundleInfo {
    pub name: String,
    pub priority: i32,
    pub adapters: Vec<String>,
    /// See [`BundleConfig::engine`] — surfaced in routing logs,
    /// metrics, and the published ``WorkItem.engine`` so workers
    /// can verify they speak the right engine.
    #[allow(dead_code)]
    pub engine: String,
}

/// Adapter-module prefixes considered legal for each engine.
///
/// The gateway's matcher intersects ``bundle.adapters`` with each
/// model's ``adapter_path`` modules — so a bundle that accidentally
/// lists an adapter outside this engine's namespace is caught at
/// config-load time rather than producing ``UnsupportedModel`` IPC
/// NAKs at runtime.
///
/// Returns an empty slice for unknown engines (caller decides what
/// to do — typically the load already rejected the bundle, so this
/// path is unreachable).
#[allow(dead_code)]
pub fn engine_adapter_prefixes(engine: &str) -> &'static [&'static str] {
    match engine {
        "pytorch" => &["sie_server.adapters."],
        "candle" => &["sie_server_rust.adapters.candle"],
        _ => &[],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bundle_config_yaml() {
        let yaml = r#"
name: default
priority: 10
adapters:
  - sie_server.adapters.sentence_transformer
  - sie_server.adapters.cross_encoder
default: true
"#;
        let config: BundleConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.name, "default");
        assert_eq!(config.priority, 10);
        assert_eq!(config.adapters.len(), 2);
        assert!(config.default);
    }

    #[test]
    fn test_bundle_config_defaults() {
        let yaml = r#"name: minimal"#;
        let config: BundleConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.priority, 100); // default_priority()
        assert!(config.adapters.is_empty());
        assert!(!config.default);
        assert!(config.machine_profiles.is_empty());
        assert_eq!(config.engine, DEFAULT_ENGINE);
    }

    #[test]
    fn test_engine_adapter_prefixes() {
        assert_eq!(
            engine_adapter_prefixes("pytorch"),
            &["sie_server.adapters."]
        );
        assert_eq!(
            engine_adapter_prefixes("candle"),
            &["sie_server_rust.adapters.candle"]
        );
        // Unknown engine → empty slice. Caller is expected to have
        // already rejected the bundle by checking ``KNOWN_ENGINES``.
        assert!(engine_adapter_prefixes("future-engine").is_empty());
        assert!(engine_adapter_prefixes("unknown").is_empty());
    }

    #[test]
    fn test_known_engines_include_candle() {
        // Lock-step parity with sie_config.model_registry.KNOWN_ENGINES.
        assert_eq!(KNOWN_ENGINES, &["pytorch", "candle"]);
    }

    #[test]
    fn sealed_is_never_a_known_engine() {
        // #1841 guard rail: "sealed" is a SERVER-DERIVED dispatch marker for org
        // custom models — it must never be a client-pinnable engine. `parse_engine_pin`
        // rejects any `X-SIE-Engine` value outside KNOWN_ENGINES with a 400, so keeping
        // "sealed" out of this list is what stops a tenant from pinning `X-SIE-Engine:
        // sealed` onto a CATALOG model and jumping into the sealed lane. If you ever add
        // "sealed" here, add a compensating gate at the engine-pin parse first.
        assert!(!KNOWN_ENGINES.contains(&"sealed"));
    }

    #[test]
    fn test_bundle_machine_profile_serde() {
        let json = r#"{"name":"l4","gpu_type":"L4","gpu_count":1,"max_batch_size":64,"max_sequence_length":512}"#;
        let profile: BundleMachineProfile = serde_json::from_str(json).unwrap();
        assert_eq!(profile.name, "l4");
        assert_eq!(profile.gpu_count, 1);

        let back = serde_json::to_string(&profile).unwrap();
        assert!(back.contains("\"name\":\"l4\""));
    }
}
