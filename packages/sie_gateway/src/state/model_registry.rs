use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use arc_swap::ArcSwap;
use sha2::{Digest, Sha256};
use tracing::{debug, error, info, warn};

use crate::types::bundle::BundleInfo;
use crate::types::model::{CanonicalProfile, ModelConfig, ModelEntry};

#[derive(Debug)]
pub struct ModelNotFoundError {
    pub model: String,
}

impl std::fmt::Display for ModelNotFoundError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Model not found: {}", self.model)
    }
}

impl std::error::Error for ModelNotFoundError {}

#[derive(Debug)]
pub struct BundleConflictError {
    pub model: String,
    pub bundle: String,
    pub compatible_bundles: Vec<String>,
}

impl std::fmt::Display for BundleConflictError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Bundle '{}' does not support model '{}'. Compatible bundles: {:?}",
            self.bundle, self.model, self.compatible_bundles
        )
    }
}

impl std::error::Error for BundleConflictError {}

#[derive(Debug)]
pub enum ResolveError {
    ModelNotFound(ModelNotFoundError),
    BundleConflict(BundleConflictError),
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolveError::ModelNotFound(e) => write!(f, "{}", e),
            ResolveError::BundleConflict(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for ResolveError {}

/// Outcome of `add_model_config`: `(created_profiles, skipped_profiles,
/// affected_bundles)`. Surfaces through to the NATS publish path so the
/// caller can log which profiles actually changed vs. were idempotently
/// skipped, and which bundles need to be re-broadcast to workers.
pub type AddModelConfigOutcome = (Vec<String>, Vec<String>, Vec<String>);

#[derive(Debug, Clone, Default)]
struct RegistrySnapshot {
    bundles: HashMap<String, BundleInfo>,
    models: HashMap<String, ModelEntry>,
    model_names_lower: HashMap<String, String>,
    bundle_config_hashes: HashMap<String, String>,
}

pub struct ModelRegistry {
    bundles_dir: PathBuf,
    models_dir: PathBuf,
    snapshot: ArcSwap<RegistrySnapshot>,
    /// Serializes the read-modify-write path. All mutators
    /// (`add_model_config`, `reload`) take this before `snapshot.load()` and
    /// release it after `snapshot.store(...)` so two concurrent writers
    /// cannot both build derived snapshots off the same base and lose each
    /// other's mutations. Readers still use `snapshot.load()` unlocked and
    /// see a consistent `Arc<RegistrySnapshot>` per access.
    write_lock: Mutex<()>,
}

impl ModelRegistry {
    pub fn new(
        bundles_dir: impl AsRef<Path>,
        models_dir: impl AsRef<Path>,
        auto_load: bool,
    ) -> Self {
        let registry = Self {
            bundles_dir: bundles_dir.as_ref().to_path_buf(),
            models_dir: models_dir.as_ref().to_path_buf(),
            snapshot: ArcSwap::from_pointee(RegistrySnapshot::default()),
            write_lock: Mutex::new(()),
        };
        if auto_load {
            registry.reload();
        }
        registry
    }

    pub fn bundles_dir(&self) -> &Path {
        &self.bundles_dir
    }

    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    fn canonical_model_name(snapshot: &RegistrySnapshot, model: &str) -> Option<String> {
        snapshot
            .model_names_lower
            .get(&model.to_lowercase())
            .cloned()
            .or_else(|| {
                if snapshot.models.contains_key(model) {
                    Some(model.to_string())
                } else {
                    None
                }
            })
    }

    pub fn reload(&self) {
        // Serialize vs add_model_config so a concurrent filesystem reload
        // cannot stomp on a partially-built snapshot.
        let _write = self
            .write_lock
            .lock()
            .expect("ModelRegistry write_lock poisoned");
        let mut new_bundles: HashMap<String, BundleInfo> = HashMap::new();
        let mut new_models: HashMap<String, ModelEntry> = HashMap::new();
        let mut new_model_names_lower: HashMap<String, String> = HashMap::new();

        // Load bundles
        if self.bundles_dir.exists() {
            match std::fs::read_dir(&self.bundles_dir) {
                Ok(entries) => {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if path.extension().and_then(|e| e.to_str()) != Some("yaml") {
                            continue;
                        }
                        match Self::load_bundle_file(&path) {
                            Ok(bundle) => {
                                debug!(
                                    bundle = %bundle.name,
                                    priority = bundle.priority,
                                    adapters = bundle.adapters.len(),
                                    "loaded bundle"
                                );
                                new_bundles.insert(bundle.name.clone(), bundle);
                            }
                            Err(e) => {
                                error!(path = %path.display(), error = %e, "failed to load bundle");
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!(dir = %self.bundles_dir.display(), error = %e, "cannot read bundles directory");
                }
            }
        } else {
            warn!(dir = %self.bundles_dir.display(), "bundles directory not found");
        }

        // Load models
        if self.models_dir.exists() {
            match std::fs::read_dir(&self.models_dir) {
                Ok(entries) => {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if !path.is_file() {
                            continue;
                        }
                        if path.extension().and_then(|e| e.to_str()) != Some("yaml") {
                            continue;
                        }
                        match Self::load_model_file(&path) {
                            Ok(model_entry) => {
                                debug!(
                                    model = %model_entry.name,
                                    adapters = ?model_entry.adapter_modules,
                                    "discovered model"
                                );
                                new_model_names_lower.insert(
                                    model_entry.name.to_lowercase(),
                                    model_entry.name.clone(),
                                );
                                new_models.insert(model_entry.name.clone(), model_entry);
                            }
                            Err(e) => {
                                error!(path = %path.display(), error = %e, "failed to load model config");
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!(dir = %self.models_dir.display(), error = %e, "cannot read models directory");
                }
            }
        } else {
            warn!(dir = %self.models_dir.display(), "models directory not found");
        }

        // Compute model-to-bundle mappings
        for model_entry in new_models.values_mut() {
            if model_entry.adapter_modules.is_empty() {
                continue;
            }
            let mut matching: Vec<(i32, String)> = Vec::new();
            for bundle in new_bundles.values() {
                let bundle_adapters: HashSet<&str> =
                    bundle.adapters.iter().map(|s| s.as_str()).collect();
                let has_overlap = model_entry
                    .adapter_modules
                    .iter()
                    .any(|a| bundle_adapters.contains(a.as_str()));
                if has_overlap {
                    matching.push((bundle.priority, bundle.name.clone()));
                }
            }
            // Break priority ties by bundle name so `model_entry.bundles[0]`
            // (the default-selected bundle at route time) is stable across
            // runs — `new_bundles` / bundle iteration comes from a
            // `HashMap`, so without a secondary key equal-priority bundles
            // would shuffle between replicas and between process restarts.
            matching.sort_by(|(pa, na), (pb, nb)| pa.cmp(pb).then_with(|| na.cmp(nb)));
            model_entry.bundles = matching.into_iter().map(|(_, name)| name).collect();
        }

        // Pre-compute bundle config hashes (expensive: sort + JSON + SHA-256).
        // Cached here so compute_bundle_config_hash is a simple HashMap lookup.
        let mut bundle_config_hashes = HashMap::new();
        for bundle_name in new_bundles.keys() {
            let hash = Self::hash_bundle_config(bundle_name, &new_bundles, &new_models);
            if !hash.is_empty() {
                bundle_config_hashes.insert(bundle_name.clone(), hash);
            }
        }

        info!(
            bundles = new_bundles.len(),
            models = new_models.len(),
            "model registry loaded"
        );

        let snap = RegistrySnapshot {
            bundles: new_bundles,
            models: new_models,
            model_names_lower: new_model_names_lower,
            bundle_config_hashes,
        };
        self.snapshot.store(Arc::new(snap));
    }

    fn load_bundle_file(path: &Path) -> Result<BundleInfo, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let data: serde_yaml::Value = serde_yaml::from_str(&content)?;

        let name = data
            .get("name")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| {
                path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string()
            });

        let priority = data.get("priority").and_then(|v| v.as_i64()).unwrap_or(100) as i32;

        let adapters = data
            .get("adapters")
            .and_then(|v| v.as_sequence())
            .map(|seq| {
                seq.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        Ok(BundleInfo {
            name,
            priority,
            adapters,
        })
    }

    fn load_model_file(path: &Path) -> Result<ModelEntry, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: ModelConfig = serde_yaml::from_str(&content)?;

        let model_name = config.name;
        let mut adapter_modules: HashSet<String> = HashSet::new();
        let mut profile_names: HashSet<String> = HashSet::new();
        let mut profile_configs: HashMap<String, CanonicalProfile> = HashMap::new();

        for (profile_name, profile) in &config.profiles {
            profile_names.insert(profile_name.clone());
            if let Some(adapter_path) = &profile.adapter_path {
                if !adapter_path.is_empty() {
                    let module = adapter_path.split(':').next().unwrap_or(adapter_path);
                    adapter_modules.insert(module.to_string());
                }
            }
            profile_configs.insert(
                profile_name.clone(),
                CanonicalProfile::from_profile(profile),
            );
        }

        Ok(ModelEntry {
            name: model_name,
            bundles: Vec::new(),
            adapter_modules,
            profile_names,
            profile_configs,
        })
    }

    pub fn resolve_bundle(
        &self,
        model: &str,
        bundle_override: Option<&str>,
    ) -> Result<String, ResolveError> {
        let snap = self.snapshot.load();

        // Case-insensitive lookup
        let canonical = Self::canonical_model_name(&snap, model);

        let canonical = match canonical {
            Some(c) => c,
            None => {
                return Err(ResolveError::ModelNotFound(ModelNotFoundError {
                    model: model.to_string(),
                }))
            }
        };

        let model_entry = match snap.models.get(&canonical) {
            Some(e) if !e.bundles.is_empty() => e,
            _ => {
                return Err(ResolveError::ModelNotFound(ModelNotFoundError {
                    model: model.to_string(),
                }))
            }
        };

        if let Some(override_bundle) = bundle_override {
            if model_entry.bundles.contains(&override_bundle.to_string()) {
                return Ok(override_bundle.to_string());
            }
            return Err(ResolveError::BundleConflict(BundleConflictError {
                model: model.to_string(),
                bundle: override_bundle.to_string(),
                compatible_bundles: model_entry.bundles.clone(),
            }));
        }

        Ok(model_entry.bundles[0].clone())
    }

    pub fn model_exists(&self, model: &str) -> bool {
        let snap = self.snapshot.load();
        snap.model_names_lower.contains_key(&model.to_lowercase())
            || snap.models.contains_key(model)
    }

    /// Fast check for "is the registry populated at all?". Used by the
    /// proxy to distinguish "unknown model in a bootstrapped registry"
    /// (→ 404) from "no models configured yet" (→ legacy fallback path).
    pub fn has_any_models(&self) -> bool {
        !self.snapshot.load().models.is_empty()
    }

    pub fn list_models(&self) -> Vec<String> {
        let snap = self.snapshot.load();
        let mut names: Vec<String> = snap.models.keys().cloned().collect();
        names.sort();
        names
    }

    /// Replace the registry's bundle set in-place, then recompute every piece
    /// of derived state that bundles feed into:
    ///
    /// - per-model `bundles` lists (which bundles a model resolves into);
    /// - per-bundle config hashes used by the worker config-skew detector.
    ///
    /// Models are left exactly as they were — only their derived bundle
    /// associations are rebuilt against the new bundle set.
    ///
    /// Used by `state::config_bootstrap` to apply bundles fetched from
    /// `sie-config`'s `GET /v1/configs/bundles` endpoint at startup, so the
    /// gateway no longer needs a filesystem seed of its own. Calling this with
    /// an empty `bundles` list is valid and clears the registry's bundle map
    /// (every subsequent `add_model_config` will then reject every adapter as
    /// unknown — that's the correct behavior when `sie-config` is unreachable
    /// and we have no seed).
    pub fn install_bundles(&self, bundles: Vec<BundleInfo>) {
        let _write = self
            .write_lock
            .lock()
            .expect("ModelRegistry write_lock poisoned");
        let old_snap = self.snapshot.load();
        let mut snap = (**old_snap).clone();

        let mut new_bundles: HashMap<String, BundleInfo> = HashMap::new();
        for bundle in bundles {
            new_bundles.insert(bundle.name.clone(), bundle);
        }

        // Rebuild model→bundle associations against the new bundle set.
        // Mirrors the post-load loop in `reload()` so a freshly-installed
        // bundle set produces identical derived state regardless of whether
        // bundles came from disk or from `sie-config`.
        for model_entry in snap.models.values_mut() {
            if model_entry.adapter_modules.is_empty() {
                model_entry.bundles.clear();
                continue;
            }
            let mut matching: Vec<(i32, String)> = Vec::new();
            for bundle in new_bundles.values() {
                let bundle_adapters: HashSet<&str> =
                    bundle.adapters.iter().map(|s| s.as_str()).collect();
                let has_overlap = model_entry
                    .adapter_modules
                    .iter()
                    .any(|a| bundle_adapters.contains(a.as_str()));
                if has_overlap {
                    matching.push((bundle.priority, bundle.name.clone()));
                }
            }
            // Break priority ties by bundle name so `model_entry.bundles[0]`
            // (the default-selected bundle at route time) is stable across
            // runs — `new_bundles` / bundle iteration comes from a
            // `HashMap`, so without a secondary key equal-priority bundles
            // would shuffle between replicas and between process restarts.
            matching.sort_by(|(pa, na), (pb, nb)| pa.cmp(pb).then_with(|| na.cmp(nb)));
            model_entry.bundles = matching.into_iter().map(|(_, name)| name).collect();
        }

        let mut bundle_config_hashes = HashMap::new();
        for bundle_name in new_bundles.keys() {
            let hash = Self::hash_bundle_config(bundle_name, &new_bundles, &snap.models);
            if !hash.is_empty() {
                bundle_config_hashes.insert(bundle_name.clone(), hash);
            }
        }

        info!(
            bundles = new_bundles.len(),
            models = snap.models.len(),
            "installed bundles into registry"
        );

        snap.bundles = new_bundles;
        snap.bundle_config_hashes = bundle_config_hashes;
        self.snapshot.store(Arc::new(snap));
    }

    pub fn list_bundles(&self) -> Vec<String> {
        let snap = self.snapshot.load();
        let mut bundles: Vec<(&String, &BundleInfo)> = snap.bundles.iter().collect();
        bundles.sort_by_key(|(_, b)| b.priority);
        bundles.into_iter().map(|(name, _)| name.clone()).collect()
    }

    pub fn get_bundle_info(&self, bundle: &str) -> Option<BundleInfo> {
        let snap = self.snapshot.load();
        snap.bundles.get(bundle).cloned()
    }

    pub fn get_model_bundles(&self, model: &str) -> Vec<String> {
        let snap = self.snapshot.load();
        let canonical = match Self::canonical_model_name(&snap, model) {
            Some(name) => name,
            None => return Vec::new(),
        };
        snap.models
            .get(&canonical)
            .map(|e| e.bundles.clone())
            .unwrap_or_default()
    }

    pub fn get_model_info(&self, model: &str) -> Option<ModelEntry> {
        let snap = self.snapshot.load();
        let canonical = Self::canonical_model_name(&snap, model)?;
        snap.models.get(&canonical).cloned()
    }

    pub fn get_model_profile_names(&self, model: &str) -> Vec<String> {
        let mut profiles: Vec<String> = self
            .get_model_info(model)
            .map(|entry| entry.profile_names.into_iter().collect())
            .unwrap_or_default();
        profiles.sort();
        profiles
    }

    /// Look up pre-computed bundle config hash (O(1) HashMap get).
    /// Hash is computed during reload() and add_model_config(), not per-request.
    pub fn compute_bundle_config_hash(&self, bundle_id: &str) -> String {
        let snap = self.snapshot.load();
        snap.bundle_config_hashes
            .get(bundle_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Compute the config hash for a bundle (expensive: sort + JSON + SHA-256).
    /// Called during reload() and add_model_config() to pre-populate the cache.
    fn hash_bundle_config(
        bundle_id: &str,
        bundles: &HashMap<String, BundleInfo>,
        models: &HashMap<String, ModelEntry>,
    ) -> String {
        let bundle = match bundles.get(bundle_id) {
            Some(b) => b,
            None => return String::new(),
        };

        let bundle_adapter_set: HashSet<&str> =
            bundle.adapters.iter().map(|s| s.as_str()).collect();

        let mut items: Vec<serde_json::Value> = Vec::new();

        let mut model_names: Vec<&String> = models.keys().collect();
        model_names.sort();

        for model_name in model_names {
            let model_entry = &models[model_name];
            if !model_entry.bundles.contains(&bundle_id.to_string()) {
                continue;
            }
            let has_overlap = model_entry
                .adapter_modules
                .iter()
                .any(|a| bundle_adapter_set.contains(a.as_str()));
            if !has_overlap {
                continue;
            }

            let mut profiles_for_hash: Vec<serde_json::Value> = Vec::new();
            let mut profile_names: Vec<&String> = model_entry.profile_names.iter().collect();
            profile_names.sort();

            for pname in profile_names {
                if let Some(p_cfg) = model_entry.profile_configs.get(pname) {
                    if let Some(ref adapter_path) = p_cfg.adapter_path {
                        let module = adapter_path.split(':').next().unwrap_or(adapter_path);
                        if !bundle_adapter_set.contains(module) {
                            continue;
                        }
                    }

                    let config = serde_json::json!({
                        "adapter_path": p_cfg.adapter_path,
                        "max_batch_tokens": p_cfg.max_batch_tokens,
                        "compute_precision": p_cfg.compute_precision,
                        "adapter_options": p_cfg.adapter_options,
                    });

                    profiles_for_hash.push(serde_json::json!({
                        "name": pname,
                        "config": config,
                    }));
                }
            }

            items.push(serde_json::json!({
                "sie_id": model_name,
                "profiles": profiles_for_hash,
            }));
        }

        if items.is_empty() {
            return String::new();
        }

        let serialized = serde_json::to_string(&items).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(serialized.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    pub fn add_model_config(&self, config: ModelConfig) -> Result<AddModelConfigOutcome, String> {
        // Serialize mutators. `ArcSwap` gives us lock-free reads but no CAS
        // on the write path — without this lock, two concurrent callers can
        // both `load()` the same base snapshot, both build a derived copy
        // with their own mutation, and the second `store()` silently drops
        // the first caller's changes. In practice this raced the NATS
        // subscription task against the bootstrap retry task.
        let _write = self
            .write_lock
            .lock()
            .expect("ModelRegistry write_lock poisoned");
        let old_snap = self.snapshot.load();
        let mut snap = (**old_snap).clone();

        let sie_id = &config.name;
        if sie_id.is_empty() {
            return Err("Missing required field: name/sie_id".to_string());
        }

        if config.profiles.is_empty() {
            return Err("Missing required field: profiles".to_string());
        }

        let mut new_adapter_modules: HashSet<String> = HashSet::new();
        for (profile_name, profile) in &config.profiles {
            if let Some(ref adapter_path) = profile.adapter_path {
                if !adapter_path.is_empty() {
                    let module = adapter_path.split(':').next().unwrap_or(adapter_path);
                    new_adapter_modules.insert(module.to_string());
                }
            } else if profile.extends.is_none() {
                return Err(format!("Profile '{}' missing adapter_path", profile_name));
            }
        }

        // Validate adapter modules are routable
        let all_bundle_adapters: HashSet<String> = snap
            .bundles
            .values()
            .flat_map(|b| b.adapters.iter().cloned())
            .collect();

        let unroutable: Vec<&String> = new_adapter_modules
            .iter()
            .filter(|a| !all_bundle_adapters.contains(a.as_str()))
            .collect();

        if !unroutable.is_empty() {
            return Err(format!(
                "Adapter(s) not in any known bundle: {}",
                unroutable
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }

        let mut created_profiles: Vec<String> = Vec::new();
        let mut skipped_profiles: Vec<String> = Vec::new();

        if let Some(existing) = snap.models.get_mut(sie_id) {
            // Append-only: add new profiles, skip identical, reject conflicts
            for (profile_name, profile) in &config.profiles {
                if existing.profile_names.contains(profile_name) {
                    let incoming = CanonicalProfile::from_profile(profile);
                    if let Some(stored) = existing.profile_configs.get(profile_name) {
                        if stored != &incoming {
                            return Err(format!(
                                "Profile '{}' on model '{}' already exists with different config (append-only)",
                                profile_name, sie_id
                            ));
                        }
                    }
                    skipped_profiles.push(profile_name.clone());
                } else {
                    created_profiles.push(profile_name.clone());
                    if let Some(ref adapter_path) = profile.adapter_path {
                        if !adapter_path.is_empty() {
                            let module = adapter_path.split(':').next().unwrap_or(adapter_path);
                            existing.adapter_modules.insert(module.to_string());
                        }
                    }
                }
            }

            for pname in &created_profiles {
                existing.profile_names.insert(pname.clone());
                if let Some(profile) = config.profiles.get(pname) {
                    existing
                        .profile_configs
                        .insert(pname.clone(), CanonicalProfile::from_profile(profile));
                }
            }
        } else {
            // New model
            let mut profile_names: HashSet<String> = HashSet::new();
            let mut profile_configs: HashMap<String, CanonicalProfile> = HashMap::new();

            for (pname, profile) in &config.profiles {
                profile_names.insert(pname.clone());
                profile_configs.insert(pname.clone(), CanonicalProfile::from_profile(profile));
            }

            created_profiles = config.profiles.keys().cloned().collect();

            snap.models.insert(
                sie_id.clone(),
                ModelEntry {
                    name: sie_id.clone(),
                    bundles: Vec::new(),
                    adapter_modules: new_adapter_modules,
                    profile_names,
                    profile_configs,
                },
            );
            snap.model_names_lower
                .insert(sie_id.to_lowercase(), sie_id.clone());
        }

        // Recompute bundle mappings for this model
        if let Some(model_entry) = snap.models.get_mut(sie_id) {
            let mut matching: Vec<(i32, String)> = Vec::new();
            for bundle in snap.bundles.values() {
                let bundle_adapters: HashSet<&str> =
                    bundle.adapters.iter().map(|s| s.as_str()).collect();
                let has_overlap = model_entry
                    .adapter_modules
                    .iter()
                    .any(|a| bundle_adapters.contains(a.as_str()));
                if has_overlap {
                    matching.push((bundle.priority, bundle.name.clone()));
                }
            }
            // Break priority ties by bundle name so `model_entry.bundles[0]`
            // (the default-selected bundle at route time) is stable across
            // runs — `new_bundles` / bundle iteration comes from a
            // `HashMap`, so without a secondary key equal-priority bundles
            // would shuffle between replicas and between process restarts.
            matching.sort_by(|(pa, na), (pb, nb)| pa.cmp(pb).then_with(|| na.cmp(nb)));
            model_entry.bundles = matching.iter().map(|(_, name)| name.clone()).collect();
        }

        let affected_bundles = snap
            .models
            .get(sie_id)
            .map(|e| e.bundles.clone())
            .unwrap_or_default();

        // Recompute config hashes for affected bundles
        for bundle_name in &affected_bundles {
            let hash = Self::hash_bundle_config(bundle_name, &snap.bundles, &snap.models);
            if hash.is_empty() {
                snap.bundle_config_hashes.remove(bundle_name);
            } else {
                snap.bundle_config_hashes.insert(bundle_name.clone(), hash);
            }
        }

        info!(
            model = %sie_id,
            created = ?created_profiles,
            skipped = ?skipped_profiles,
            bundles = ?affected_bundles,
            "added model config"
        );

        self.snapshot.store(Arc::new(snap));

        Ok((created_profiles, skipped_profiles, affected_bundles))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn create_test_dirs() -> (TempDir, PathBuf, PathBuf) {
        let dir = TempDir::new().unwrap();
        let bundles_dir = dir.path().join("bundles");
        let models_dir = dir.path().join("models");
        fs::create_dir_all(&bundles_dir).unwrap();
        fs::create_dir_all(&models_dir).unwrap();
        (dir, bundles_dir, models_dir)
    }

    #[test]
    fn test_empty_registry() {
        let (_dir, bundles_dir, models_dir) = create_test_dirs();
        let registry = ModelRegistry::new(&bundles_dir, &models_dir, true);
        assert!(registry.list_models().is_empty());
        assert!(registry.list_bundles().is_empty());
        assert!(!registry.has_any_models());
    }

    #[test]
    fn test_has_any_models_flips_after_add() {
        use crate::types::model::{ModelConfig, ProfileConfig};
        use std::collections::HashMap as StdHashMap;

        let (_dir, bundles_dir, models_dir) = create_test_dirs();
        fs::write(
            bundles_dir.join("default.yaml"),
            "name: default\npriority: 10\nadapters:\n  - sie_server.adapters.sentence_transformer\n",
        )
        .unwrap();
        let registry = ModelRegistry::new(&bundles_dir, &models_dir, true);
        assert!(
            !registry.has_any_models(),
            "empty registry must report no models"
        );

        let mut profiles = StdHashMap::new();
        profiles.insert(
            "default".to_string(),
            ProfileConfig {
                adapter_path: Some("sie_server.adapters.sentence_transformer:Adapter".to_string()),
                max_batch_tokens: Some(4096),
                compute_precision: None,
                adapter_options: None,
                extends: None,
            },
        );
        registry
            .add_model_config(ModelConfig {
                name: "org/x".to_string(),
                adapter_module: None,
                default_bundle: None,
                profiles,
            })
            .unwrap();
        assert!(registry.has_any_models(), "must report populated after add");
    }

    #[test]
    fn test_concurrent_add_model_config_does_not_lose_updates() {
        // Guards the `write_lock` around `snapshot.load() + clone() +
        // store()` in `add_model_config`. Without the lock, concurrent
        // writers can both load the same base snapshot and one of the
        // updates gets silently dropped. Every model added by every worker
        // must end up in the registry.
        use crate::types::model::{ModelConfig, ProfileConfig};
        use std::collections::HashMap as StdHashMap;
        use std::thread;

        let (_dir, bundles_dir, models_dir) = create_test_dirs();
        fs::write(
            bundles_dir.join("default.yaml"),
            "name: default\npriority: 10\nadapters:\n  - sie_server.adapters.sentence_transformer\n",
        )
        .unwrap();
        let registry = Arc::new(ModelRegistry::new(&bundles_dir, &models_dir, true));

        let workers = 16;
        let adds_per_worker = 25;
        let mut handles = Vec::with_capacity(workers);
        for w in 0..workers {
            let registry = Arc::clone(&registry);
            handles.push(thread::spawn(move || {
                for i in 0..adds_per_worker {
                    let name = format!("w{}/m{}", w, i);
                    let mut profiles = StdHashMap::new();
                    profiles.insert(
                        "default".to_string(),
                        ProfileConfig {
                            adapter_path: Some(
                                "sie_server.adapters.sentence_transformer:Adapter".to_string(),
                            ),
                            max_batch_tokens: Some(4096),
                            compute_precision: None,
                            adapter_options: None,
                            extends: None,
                        },
                    );
                    let cfg = ModelConfig {
                        name: name.clone(),
                        adapter_module: None,
                        default_bundle: None,
                        profiles,
                    };
                    registry.add_model_config(cfg).expect("add must succeed");
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        let models = registry.list_models();
        assert_eq!(
            models.len(),
            workers * adds_per_worker,
            "every parallel insert must be reflected in the registry"
        );
    }

    #[test]
    fn test_load_bundle() {
        let (_dir, bundles_dir, models_dir) = create_test_dirs();

        fs::write(
            bundles_dir.join("default.yaml"),
            r#"
name: default
priority: 10
adapters:
  - sie_server.adapters.sentence_transformer
  - sie_server.adapters.cross_encoder
default: true
"#,
        )
        .unwrap();

        let registry = ModelRegistry::new(&bundles_dir, &models_dir, true);
        let bundles = registry.list_bundles();
        assert_eq!(bundles.len(), 1);
        assert_eq!(bundles[0], "default");

        let info = registry.get_bundle_info("default").unwrap();
        assert_eq!(info.priority, 10);
        assert_eq!(info.adapters.len(), 2);
    }

    #[test]
    fn test_load_model_and_resolve() {
        let (_dir, bundles_dir, models_dir) = create_test_dirs();

        fs::write(
            bundles_dir.join("default.yaml"),
            r#"
name: default
priority: 10
adapters:
  - sie_server.adapters.sentence_transformer
"#,
        )
        .unwrap();

        fs::write(
            models_dir.join("bge-m3.yaml"),
            r#"
name: BAAI/bge-m3
profiles:
  default:
    adapter_path: "sie_server.adapters.sentence_transformer:SentenceTransformerAdapter"
    max_batch_tokens: 4096
"#,
        )
        .unwrap();

        let registry = ModelRegistry::new(&bundles_dir, &models_dir, true);
        assert!(registry.model_exists("BAAI/bge-m3"));
        assert!(registry.model_exists("baai/bge-m3")); // case insensitive

        let bundle = registry.resolve_bundle("BAAI/bge-m3", None).unwrap();
        assert_eq!(bundle, "default");
    }

    #[test]
    fn test_resolve_model_not_found() {
        let (_dir, bundles_dir, models_dir) = create_test_dirs();
        let registry = ModelRegistry::new(&bundles_dir, &models_dir, true);

        let result = registry.resolve_bundle("nonexistent/model", None);
        assert!(matches!(result, Err(ResolveError::ModelNotFound(_))));
    }

    #[test]
    fn test_resolve_bundle_conflict() {
        let (_dir, bundles_dir, models_dir) = create_test_dirs();

        fs::write(
            bundles_dir.join("default.yaml"),
            r#"
name: default
priority: 10
adapters:
  - sie_server.adapters.sentence_transformer
"#,
        )
        .unwrap();

        fs::write(
            models_dir.join("bge-m3.yaml"),
            r#"
name: BAAI/bge-m3
profiles:
  default:
    adapter_path: "sie_server.adapters.sentence_transformer:SentenceTransformerAdapter"
"#,
        )
        .unwrap();

        let registry = ModelRegistry::new(&bundles_dir, &models_dir, true);
        let result = registry.resolve_bundle("BAAI/bge-m3", Some("nonexistent"));
        assert!(matches!(result, Err(ResolveError::BundleConflict(_))));
    }

    #[test]
    fn test_bundle_priority_ordering() {
        let (_dir, bundles_dir, models_dir) = create_test_dirs();

        fs::write(
            bundles_dir.join("high.yaml"),
            r#"
name: high
priority: 50
adapters:
  - sie_server.adapters.sentence_transformer
"#,
        )
        .unwrap();

        fs::write(
            bundles_dir.join("low.yaml"),
            r#"
name: low
priority: 5
adapters:
  - sie_server.adapters.sentence_transformer
"#,
        )
        .unwrap();

        fs::write(
            models_dir.join("bge-m3.yaml"),
            r#"
name: BAAI/bge-m3
profiles:
  default:
    adapter_path: "sie_server.adapters.sentence_transformer:SentenceTransformerAdapter"
"#,
        )
        .unwrap();

        let registry = ModelRegistry::new(&bundles_dir, &models_dir, true);

        // "low" has priority 5, "high" has 50. Lower priority number = higher priority.
        let bundle = registry.resolve_bundle("BAAI/bge-m3", None).unwrap();
        assert_eq!(bundle, "low");
    }

    #[test]
    fn test_add_model_config() {
        let (_dir, bundles_dir, models_dir) = create_test_dirs();

        fs::write(
            bundles_dir.join("default.yaml"),
            r#"
name: default
priority: 10
adapters:
  - sie_server.adapters.sentence_transformer
"#,
        )
        .unwrap();

        let registry = ModelRegistry::new(&bundles_dir, &models_dir, true);

        let config = ModelConfig {
            name: "test/model".to_string(),
            adapter_module: None,
            default_bundle: None,
            profiles: {
                let mut m = HashMap::new();
                m.insert(
                    "default".to_string(),
                    crate::types::model::ProfileConfig {
                        adapter_path: Some(
                            "sie_server.adapters.sentence_transformer:SentenceTransformerAdapter"
                                .to_string(),
                        ),
                        max_batch_tokens: Some(4096),
                        compute_precision: None,
                        adapter_options: None,
                        extends: None,
                    },
                );
                m
            },
        };

        let (created, skipped, bundles) = registry.add_model_config(config).unwrap();
        assert_eq!(created, vec!["default"]);
        assert!(skipped.is_empty());
        assert_eq!(bundles, vec!["default"]);

        assert!(registry.model_exists("test/model"));
        let resolved = registry.resolve_bundle("test/model", None).unwrap();
        assert_eq!(resolved, "default");
    }

    #[test]
    fn test_compute_bundle_config_hash() {
        let (_dir, bundles_dir, models_dir) = create_test_dirs();

        fs::write(
            bundles_dir.join("default.yaml"),
            r#"
name: default
priority: 10
adapters:
  - sie_server.adapters.sentence_transformer
"#,
        )
        .unwrap();

        fs::write(
            models_dir.join("bge-m3.yaml"),
            r#"
name: BAAI/bge-m3
profiles:
  default:
    adapter_path: "sie_server.adapters.sentence_transformer:SentenceTransformerAdapter"
    max_batch_tokens: 4096
"#,
        )
        .unwrap();

        let registry = ModelRegistry::new(&bundles_dir, &models_dir, true);
        let hash = registry.compute_bundle_config_hash("default");
        assert!(!hash.is_empty());
        assert_eq!(hash.len(), 64); // SHA-256 hex

        // Hash should be deterministic
        let hash2 = registry.compute_bundle_config_hash("default");
        assert_eq!(hash, hash2);

        // Unknown bundle returns empty
        let empty = registry.compute_bundle_config_hash("nonexistent");
        assert!(empty.is_empty());
    }

    #[test]
    fn test_concurrent_add_model_config_preserves_all_writes() {
        // Fix #4 regression: without `write_lock`, two threads that call
        // `add_model_config` in parallel can both `snapshot.load()` the
        // same base, both build a derived clone with only their own
        // model, and the second `store()` silently clobbers the first.
        // This test fires 32 threads at the same registry and asserts
        // every model ended up visible.
        use std::sync::Arc;
        use std::thread;

        let (_dir, bundles_dir, models_dir) = create_test_dirs();
        fs::write(
            bundles_dir.join("default.yaml"),
            r#"
name: default
priority: 10
adapters:
  - sie_server.adapters.sentence_transformer
"#,
        )
        .unwrap();
        let registry = Arc::new(ModelRegistry::new(&bundles_dir, &models_dir, true));

        let n: usize = 32;
        let mut handles = Vec::with_capacity(n);
        for i in 0..n {
            let reg = Arc::clone(&registry);
            handles.push(thread::spawn(move || {
                let config = ModelConfig {
                    name: format!("race/model-{i}"),
                    adapter_module: None,
                    default_bundle: None,
                    profiles: {
                        let mut m = HashMap::new();
                        m.insert(
                            "default".to_string(),
                            crate::types::model::ProfileConfig {
                                adapter_path: Some(
                                    "sie_server.adapters.sentence_transformer:A".to_string(),
                                ),
                                max_batch_tokens: Some(4096),
                                compute_precision: None,
                                adapter_options: None,
                                extends: None,
                            },
                        );
                        m
                    },
                };
                reg.add_model_config(config).expect("add_model_config");
            }));
        }
        for h in handles {
            h.join().expect("thread join");
        }

        for i in 0..n {
            let id = format!("race/model-{i}");
            assert!(
                registry.model_exists(&id),
                "{id} should be present after concurrent adds; if this fails \
                 the ArcSwap load-clone-store is racing (bug #4 regression)."
            );
        }
    }

    #[test]
    fn test_reload() {
        let (_dir, bundles_dir, models_dir) = create_test_dirs();

        let registry = ModelRegistry::new(&bundles_dir, &models_dir, true);
        assert!(registry.list_models().is_empty());

        // Add files and reload
        fs::write(
            bundles_dir.join("default.yaml"),
            r#"
name: default
priority: 10
adapters:
  - sie_server.adapters.sentence_transformer
"#,
        )
        .unwrap();

        fs::write(
            models_dir.join("model.yaml"),
            r#"
name: test/model
profiles:
  default:
    adapter_path: "sie_server.adapters.sentence_transformer:Adapter"
"#,
        )
        .unwrap();

        registry.reload();
        assert_eq!(registry.list_models().len(), 1);
        assert_eq!(registry.list_bundles().len(), 1);
    }
}
