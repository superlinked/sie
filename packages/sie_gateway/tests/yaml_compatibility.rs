use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};
use serde_saphyr as yaml;
use sie_gateway::state::model_registry::ModelRegistry;
use sie_gateway::types::model::{ModelConfig, ModelEntry};

fn yaml_files(directory: &Path) -> Vec<PathBuf> {
    let mut files: Vec<_> = std::fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            matches!(
                path.extension().and_then(|ext| ext.to_str()),
                Some("yaml" | "yml")
            )
        })
        .collect();
    files.sort();
    assert!(
        !files.is_empty(),
        "no YAML files in {}",
        directory.display()
    );
    files
}

fn entry_snapshot(entry: &ModelEntry) -> Value {
    let extras = &entry.info_extras;
    let profiles: Map<_, _> = entry
        .profile_configs
        .iter()
        .map(|(name, profile)| {
            (
                name.clone(),
                json!({
                    "kv_budget_tokens": profile.kv_budget_tokens,
                    "adapter_path": profile.adapter_path,
                    "max_batch_tokens": profile.max_batch_tokens,
                    "compute_precision": profile.compute_precision,
                    "max_output_tokens": profile.max_output_tokens,
                    "adapter_options": profile.adapter_options,
                    "grammar_profile": profile.grammar_profile,
                    "chat_template_kwargs": profile.chat_template_kwargs,
                }),
            )
        })
        .collect();
    json!({
        "name": entry.name,
        "canonical_base_model": entry.canonical_base_model,
        "canonical_profile": entry.canonical_profile,
        "pool": entry.pool,
        "bundles": entry.bundles.iter().collect::<BTreeSet<_>>(),
        "adapter_modules": entry.adapter_modules.iter().collect::<BTreeSet<_>>(),
        "profile_names": entry.profile_names.iter().collect::<BTreeSet<_>>(),
        "profile_configs": profiles,
        "extras": {
            "inputs": extras.inputs.iter().collect::<BTreeSet<_>>(),
            "outputs": extras.outputs,
            "dims": extras.dims,
            "max_sequence_length": extras.max_sequence_length,
            "revision": extras.revision,
            "routing": extras.routing,
            "max_output_tokens": extras.max_output_tokens,
            "profile_max_output_tokens": extras.profile_max_output_tokens,
            "grammar_capabilities": extras.grammar_capabilities,
            "streaming_supported": extras.streaming_supported,
            "grammar_profile": extras.grammar_profile,
            "profile_parents": extras.profile_parents,
            "tools_supported": extras.tools_supported,
            "code": extras.code,
            "sql": extras.sql,
            "guard": extras.guard,
            "lora_adapters": extras.lora_adapters.as_ref().map(|names| names.iter().collect::<BTreeSet<_>>()),
            "profile_lora_adapters": extras.profile_lora_adapters,
        },
    })
}

fn registry_snapshot(registry: &ModelRegistry) -> Value {
    let models: Map<_, _> = registry
        .list_models()
        .into_iter()
        .map(|name| {
            let entry = registry.get_model_info(&name).unwrap();
            (name, entry_snapshot(&entry))
        })
        .collect();
    let bundles: Map<_, _> = registry
        .list_bundles()
        .into_iter()
        .map(|name| {
            let bundle = registry.get_bundle_info(&name).unwrap();
            let pools: BTreeSet<_> = registry
                .list_models()
                .iter()
                .filter_map(|model| registry.get_model_pool_name(model))
                .collect();
            let pool_hashes: Map<_, _> = pools
                .into_iter()
                .map(|pool| {
                    let hash = registry.compute_bundle_config_hash_for_pool(&name, &pool);
                    (pool, json!(hash))
                })
                .collect();
            let value = json!({
                "name": bundle.name,
                "priority": bundle.priority,
                "adapters": bundle.adapters,
                "engine": bundle.engine,
                "hash": registry.compute_bundle_config_hash(&name),
                "pool_hashes": pool_hashes,
            });
            (name, value)
        })
        .collect();
    json!({"models": models, "bundles": bundles})
}

#[test]
fn catalog_yaml_compatibility() {
    let root = std::env::var_os("SIE_YAML_CATALOG_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../sie_server"));
    let bundles = root.join("bundles");
    let models = root.join("models");
    let bundle_files = yaml_files(&bundles);
    let model_files = yaml_files(&models);
    let mut documents = Map::new();
    for (kind, files) in [("bundles", &bundle_files), ("models", &model_files)] {
        for path in files {
            let source = std::fs::read_to_string(path).unwrap();
            let raw: Value = yaml::from_str(&source)
                .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
            let exported = yaml::to_string(&raw).unwrap();
            assert_eq!(
                raw,
                yaml::from_str::<Value>(&exported).unwrap(),
                "{}",
                path.display()
            );
            documents.insert(
                format!("{kind}/{}", path.file_name().unwrap().to_str().unwrap()),
                raw,
            );
        }
    }

    let mut per_file = Map::new();
    for path in &model_files {
        let source = std::fs::read_to_string(path).unwrap();
        let config: ModelConfig =
            yaml::from_str(&source).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        let single_model = tempfile::tempdir().unwrap();
        std::fs::write(single_model.path().join("model.yaml"), &source).unwrap();
        let filesystem =
            ModelRegistry::with_threshold_routing(&bundles, single_model.path(), true, true);
        assert!(
            filesystem.has_any_models(),
            "{} was skipped",
            path.display()
        );
        assert_eq!(filesystem.list_bundles().len(), bundle_files.len());

        let empty_models = tempfile::tempdir().unwrap();
        let delta =
            ModelRegistry::with_threshold_routing(&bundles, empty_models.path(), true, true);
        delta
            .add_model_config(config)
            .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        let snapshot = json!({
            "filesystem": registry_snapshot(&filesystem),
            "delta": registry_snapshot(&delta),
        });
        per_file.insert(
            path.file_name().unwrap().to_str().unwrap().to_owned(),
            snapshot,
        );
    }
    let full = ModelRegistry::with_threshold_routing(&bundles, &models, true, true);
    let snapshot =
        json!({"documents": documents, "per_file": per_file, "combined": registry_snapshot(&full)});
    if let Some(path) = std::env::var_os("SIE_YAML_SNAPSHOT") {
        std::fs::write(path, serde_json::to_vec_pretty(&snapshot).unwrap()).unwrap();
    }
    println!(
        "validated {} model files and {} bundle files",
        model_files.len(),
        bundle_files.len()
    );
}
