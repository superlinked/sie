use serde::{Deserialize, Serialize};

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
