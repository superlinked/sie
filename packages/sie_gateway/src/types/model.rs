use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::{HashMap, HashSet};

/// Fields mirrored from model YAML for ``/v1/models`` wire parity with
/// ``sie_server`` ``ModelInfo`` (inputs, outputs, dims, profiles, …).
#[derive(Debug, Clone, Default)]
pub struct ModelInfoExtras {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub dims: HashMap<String, i64>,
    pub max_sequence_length: Option<u64>,
    /// Immutable Hugging Face commit SHA for the model weights. This is part
    /// of worker/gateway parity and is absent for unpinned or package-backed
    /// models.
    pub revision: Option<String>,
    /// Per-request hard cap on ``max_new_tokens``. Mirrors
    /// ``tasks.generate.max_output_tokens`` from the model YAML; absent
    /// when the model has no ``generate`` task or the field was not
    /// set.
    pub max_output_tokens: Option<u32>,
    /// ``tasks.generate.capabilities.grammar`` from the
    /// model YAML — the list of grammar ``kind`` strings the model
    /// supports (e.g. ``["json_schema", "regex"]``). Gateway rejects
    /// requests whose ``grammar.kind`` is not in this list with 400
    /// ``unsupported_field``. ``None`` when the model has no
    /// ``generate`` task; empty ``Vec`` when grammar is explicitly
    /// disabled.
    pub grammar_capabilities: Option<Vec<String>>,
    /// ``tasks.generate.grammar_profile`` from the model YAML — the name of a
    /// profile that grammar-constrained requests must be served on. When set,
    /// the chat/completions/generate handlers rewrite a grammar request's model
    /// id to the ``{model}:{grammar_profile}`` variant so it runs on a profile
    /// that enforces the grammar correctly (e.g. ``no-spec`` — speculative
    /// decoding bypasses SGLang's Outlines FSM). ``None`` when the model has no
    /// ``generate`` task or does not declare the field (no rewrite).
    pub grammar_profile: Option<String>,
    /// ``tasks.generate.capabilities.tools`` from the model YAML — a
    /// single boolean advertising whether the model supports
    /// OpenAI-style tool calling. The chat completion handler gates
    /// requests carrying a ``tools`` array on this flag with a 400
    /// ``unsupported_field`` when the model declares ``false`` (or
    /// has no ``generate`` task). Defaults to ``false`` per
    /// :class:`sie_server.config.model.GenerateCapabilities`.
    pub tools_supported: Option<bool>,
    /// ``tasks.generate.capabilities.code`` from the model YAML — a
    /// boolean advertising whether the model is validated for code
    /// generation (backs the ``model="code"`` alias). Informational
    /// only; never request-gated. Defaults to ``false`` when the
    /// ``capabilities`` block or the key is absent.
    pub code: bool,
    /// ``tasks.generate.capabilities.sql`` from the model YAML — a
    /// boolean advertising whether the model targets the SQL path
    /// (backs the ``model="sql"`` alias). Informational only; never
    /// request-gated. Defaults to ``false`` when absent.
    pub sql: bool,
    /// ``tasks.generate.capabilities.guard`` from the model YAML — a
    /// boolean advertising whether the model is a content-moderation /
    /// policy-check guard (backs the ``model="guard"`` alias).
    /// Informational only; never request-gated. Defaults to ``false``
    /// when absent.
    pub guard: bool,
    /// Multi-LoRA served-names advertised on ``/v1/models`` under
    /// ``capabilities.lora_adapters``. Union of the public served-names across
    /// the model's profiles (``profiles.<p>.adapter_options.loadtime.lora_paths``
    /// keys). Only served-names are surfaced — never the source HF-ids/paths
    /// (avoids leaking a tenant's private adapter source). ``None`` when the
    /// model declares no adapters. Kept for back-compat as the model-level
    /// summary; per-profile precise scoping lives in ``profile_lora_adapters``.
    pub lora_adapters: Option<Vec<String>>,
    /// Per-profile breakdown of the served-names advertised in
    /// ``lora_adapters``. Each entry is ``profile_name → [adapter_name,
    /// ...]`` — the set of LoRA served-names actually configured for that
    /// profile (``profiles.<p>.adapter_options.loadtime.lora_paths`` keys).
    /// ``None`` when the model declares no adapters on any profile. Profiles
    /// that exist but declare zero adapters are omitted (the validation
    /// path treats a missing entry the same as "no adapters advertised
    /// here"). Validation MUST use this map — not ``lora_adapters`` — so a
    /// request for profile A cannot accept an adapter only configured for
    /// profile B. ``lora_adapters`` (the union) is derivable by flattening
    /// the values and deduping.
    pub profile_lora_adapters: Option<HashMap<String, Vec<String>>>,
}

impl ModelInfoExtras {
    /// Best-effort extraction from a raw model YAML document (same files as
    /// ``sie_server`` / ``sie-config``). Missing sections fall back to
    /// conservative defaults so the JSON shape stays valid.
    pub fn from_yaml_raw(raw: &serde_yaml::Value) -> Self {
        let mut extras = Self::default();

        if let serde_yaml::Value::Mapping(m) = raw.get("inputs").unwrap_or(&serde_yaml::Value::Null)
        {
            for (k, v) in m {
                let Some(ks) = k.as_str() else {
                    continue;
                };
                if matches!(v, serde_yaml::Value::Bool(true)) {
                    extras.inputs.push(ks.to_string());
                }
            }
        }
        if extras.inputs.is_empty() {
            extras.inputs.push("text".to_string());
        }

        extras.max_sequence_length = raw.get("max_sequence_length").and_then(|v| v.as_u64());
        extras.revision = raw
            .get("hf_revision")
            .and_then(|value| value.as_str())
            .filter(|value| !value.is_empty())
            .map(str::to_owned);

        let tasks = raw.get("tasks");
        if let Some(enc) = tasks.and_then(|t| match t.get("encode")? {
            serde_yaml::Value::Mapping(m) => Some(m),
            _ => None,
        }) {
            for (k, v) in enc {
                let Some(key) = k.as_str() else {
                    continue;
                };
                if key.is_empty() {
                    continue;
                }
                match v {
                    serde_yaml::Value::Mapping(vm) => {
                        if let Some(dim) = vm.get("dim").and_then(|d| {
                            d.as_u64().or_else(|| d.as_i64().map(|i| i.max(0) as u64))
                        }) {
                            extras.dims.insert(key.to_string(), dim as i64);
                            extras.outputs.push(key.to_string());
                        } else if !vm.is_empty() {
                            extras.outputs.push(key.to_string());
                        }
                    }
                    serde_yaml::Value::Null => {}
                    _ => {
                        extras.outputs.push(key.to_string());
                    }
                }
            }
        }

        // Non-encode tasks contribute output kinds too, mirroring
        // ``ModelConfig.outputs`` in ``packages/sie_server/src/sie_server/config/model.py``.
        if let Some(t) = tasks {
            if matches!(t.get("score"), Some(serde_yaml::Value::Mapping(_))) {
                extras.outputs.push("score".to_string());
            }
            if matches!(t.get("extract"), Some(serde_yaml::Value::Mapping(_))) {
                extras.outputs.push("json".to_string());
            }
            if let Some(serde_yaml::Value::Mapping(gen)) = t.get("generate") {
                extras.outputs.push("tokens".to_string());
                // Surface the per-request cap so
                // ``proxy_chat`` can reject overflowing
                // ``max_completion_tokens`` requests pre-publish.
                if let Some(cap) = gen
                    .get("max_output_tokens")
                    .and_then(|v| v.as_u64())
                    .and_then(|n| u32::try_from(n).ok())
                {
                    extras.max_output_tokens = Some(cap);
                }
                // Read the per-model grammar capability list.
                // The YAML shape is
                // ``tasks.generate.capabilities.grammar: [kind, ...]``;
                // missing => default to an empty list so the gateway
                // explicitly rejects grammar requests rather than
                // silently passing them through. A missing
                // ``generate`` task leaves ``grammar_capabilities ==
                // None`` so non-generation endpoints don't fail the
                // gate.
                let capabilities = gen.get("capabilities").and_then(|c| match c {
                    serde_yaml::Value::Mapping(m) => Some(m),
                    _ => None,
                });
                let grammar = capabilities
                    .and_then(|m| m.get("grammar"))
                    .and_then(|g| match g {
                        serde_yaml::Value::Sequence(seq) => Some(
                            seq.iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect::<Vec<String>>(),
                        ),
                        _ => None,
                    })
                    .unwrap_or_default();
                extras.grammar_capabilities = Some(grammar);
                // ``tasks.generate.grammar_profile: <name>`` — optional profile
                // that grammar requests are routed to (the gateway rewrites the
                // model id to ``{model}:{name}``). Absent => no rewrite.
                extras.grammar_profile = gen
                    .get("grammar_profile")
                    .and_then(|v| v.as_str())
                    .map(String::from);
                // ``capabilities.tools: bool`` defaults to ``false``
                // — mirrors :class:`GenerateCapabilities` in the
                // worker config. Absent ``capabilities`` block also
                // resolves to ``false``.
                let tools = capabilities
                    .and_then(|m| m.get("tools"))
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                extras.tools_supported = Some(tools);
                // ``capabilities.{code,sql,guard}: bool`` — informational
                // job-capability flags that back the ``model="code"`` /
                // ``"sql"`` / ``"guard"`` aliases. Each defaults to
                // ``false`` when the ``capabilities`` block or the key is
                // absent (mirrors :class:`GenerateCapabilities`).
                extras.code = capabilities
                    .and_then(|m| m.get("code"))
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                extras.sql = capabilities
                    .and_then(|m| m.get("sql"))
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                extras.guard = capabilities
                    .and_then(|m| m.get("guard"))
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
            }
        }

        // Multi-LoRA: collect the served-names declared per profile and
        // also flatten/dedup them into the model-level union. The
        // per-profile map is the precise capability the gateway gates
        // against; the union is preserved for back-compat on /v1/models.
        // Only the public served-names are advertised — never the source
        // HF-ids/paths.
        if let Some(serde_yaml::Value::Mapping(profiles)) = raw.get("profiles") {
            let mut per_profile: HashMap<String, Vec<String>> = HashMap::new();
            let mut union_names: Vec<String> = Vec::new();
            for (pname, pcfg) in profiles {
                let Some(profile_name) = pname.as_str() else {
                    continue;
                };
                let lora_paths = pcfg
                    .get("adapter_options")
                    .and_then(|a| a.get("loadtime"))
                    .and_then(|l| l.get("lora_paths"));
                let mut profile_names: Vec<String> = Vec::new();
                if let Some(serde_yaml::Value::Mapping(m)) = lora_paths {
                    for k in m.keys() {
                        if let Some(s) = k.as_str() {
                            if !profile_names.iter().any(|n| n == s) {
                                profile_names.push(s.to_string());
                            }
                            if !union_names.iter().any(|n| n == s) {
                                union_names.push(s.to_string());
                            }
                        }
                    }
                }
                if !profile_names.is_empty() {
                    per_profile.insert(profile_name.to_string(), profile_names);
                }
            }
            if !union_names.is_empty() {
                extras.lora_adapters = Some(union_names);
            }
            if !per_profile.is_empty() {
                extras.profile_lora_adapters = Some(per_profile);
            }
        }

        let tasks_absent_or_empty = match tasks {
            None | Some(serde_yaml::Value::Null) => true,
            Some(serde_yaml::Value::Mapping(m)) => m.is_empty(),
            _ => false,
        };
        if extras.outputs.is_empty() && tasks_absent_or_empty {
            extras.outputs.push("dense".to_string());
        }

        extras
    }

    pub fn from_model_config(config: &ModelConfig) -> Self {
        match serde_yaml::to_value(config) {
            Ok(v) => Self::from_yaml_raw(&v),
            Err(_) => Self {
                inputs: vec!["text".to_string()],
                outputs: vec!["dense".to_string()],
                dims: HashMap::new(),
                max_sequence_length: config.max_sequence_length,
                revision: None,
                max_output_tokens: None,
                grammar_capabilities: None,
                grammar_profile: None,
                tools_supported: None,
                code: false,
                sql: false,
                guard: false,
                lora_adapters: None,
                profile_lora_adapters: None,
            },
        }
    }

    /// Whether the model can serve vision *generation* on
    /// ``/v1/chat/completions``: it must accept image input AND declare a
    /// generation task. ``inputs.image`` alone is also set by encode-only
    /// image models (CLIP/SigLIP), which must not accept a chat image
    /// request; ``outputs`` carries ``"tokens"`` only when a ``generate``
    /// task is present (see ``from_yaml_raw``).
    pub fn supports_vision_generation(&self) -> bool {
        self.inputs.iter().any(|s| s == "image") && self.outputs.iter().any(|s| s == "tokens")
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelConfig {
    #[serde(alias = "sie_id")]
    pub name: String,
    #[serde(default)]
    pub hf_revision: Option<String>,
    #[serde(default)]
    pub adapter_module: Option<String>,
    #[serde(default)]
    pub default_bundle: Option<String>,
    #[serde(default)]
    pub pool: Option<String>,
    #[serde(default)]
    pub profiles: HashMap<String, ProfileConfig>,
    /// Model YAML ``inputs:`` map (e.g. ``text: true``).
    #[serde(default)]
    pub inputs: Option<HashMap<String, bool>>,
    #[serde(default)]
    pub max_sequence_length: Option<u64>,
    /// Full ``tasks:`` tree from YAML (encode outputs / dims).
    #[serde(default)]
    pub tasks: Option<serde_yaml::Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ProfileConfig {
    #[serde(default)]
    pub adapter_path: Option<String>,
    #[serde(default)]
    pub max_batch_tokens: Option<u32>,
    #[serde(default)]
    pub compute_precision: Option<String>,
    #[serde(default)]
    pub adapter_options: Option<serde_json::Value>,
    #[serde(default)]
    pub extends: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ModelEntry {
    pub name: String,
    pub pool: Option<String>,
    pub bundles: Vec<String>,
    pub adapter_modules: HashSet<String>,
    pub profile_names: HashSet<String>,
    pub profile_configs: HashMap<String, CanonicalProfile>,
    pub info_extras: ModelInfoExtras,
}

impl ModelEntry {
    /// JSON shaped like ``sie_server.api.models.ModelInfo`` for HTTP clients.
    pub fn to_model_info_value(&self, loaded: bool) -> Value {
        let state = if loaded { "loaded" } else { "available" };
        let mut profiles = Map::new();
        for pname in &self.profile_names {
            profiles.insert(pname.clone(), json!({ "is_default": pname == "default" }));
        }
        json!({
            "name": self.name,
            "inputs": self.info_extras.inputs,
            "outputs": self.info_extras.outputs,
            "dims": self.info_extras.dims,
            "loaded": loaded,
            "state": state,
            "last_error": Value::Null,
            "max_sequence_length": self.info_extras.max_sequence_length,
            "revision": self.info_extras.revision,
            "profiles": Value::Object(profiles),
            // Advertised model capabilities. ``lora_adapters`` lists the
            // public served-names of declared LoRA adapters (union across
            // profiles, never source paths; omitted when the model declares
            // none). ``profile_lora_adapters`` is the per-profile breakdown
            // — added by G-M10 so consumers needing precise routing scope
            // don't have to reverse-engineer it from the union. Both
            // fields are documented in the OpenAPI schema on
            // ``ModelCapabilitiesWire``.
            "capabilities": {
                "lora_adapters": self.info_extras.lora_adapters,
                "profile_lora_adapters": self.info_extras.profile_lora_adapters,
                "grammar": self.info_extras.grammar_capabilities,
                "tools": self.info_extras.tools_supported,
                "code": self.info_extras.code,
                "sql": self.info_extras.sql,
                "guard": self.info_extras.guard,
            },
        })
    }

    /// Per-profile LoRA-adapter served-names for this entry, scoped to a
    /// specific profile (not the union across profiles).
    ///
    /// Validation MUST go through here, not ``info_extras.lora_adapters``
    /// — the union accepts an adapter that's only configured for a
    /// *different* profile, so the gateway would pass through a request
    /// the worker then fails with an opaque "adapter not loaded" instead
    /// of a clean 400 ``unknown_lora_adapter``. Closes review finding
    /// M10.
    ///
    /// Returns:
    /// * ``Some(&Vec<String>)`` — the model advertises adapters for this
    ///   profile; validate the request against this list.
    /// * ``None`` — either the model declares no LoRA adapters at all, or
    ///   the given profile exists but has no adapters configured. In
    ///   either case the gate should reject any non-``None`` request
    ///   ``lora_adapter`` because nothing is advertised here.
    pub fn lora_adapters_for_profile(&self, profile_name: &str) -> Option<&Vec<String>> {
        self.info_extras
            .profile_lora_adapters
            .as_ref()
            .and_then(|m| m.get(profile_name))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CanonicalProfile {
    pub adapter_path: Option<String>,
    pub max_batch_tokens: Option<u32>,
    pub compute_precision: Option<String>,
    pub adapter_options: Option<serde_json::Value>,
}

impl CanonicalProfile {
    pub fn from_profile(profile: &ProfileConfig) -> Self {
        // Python-compat normalization. `sie_config.model_registry`
        // canonicalizes `adapter_options` via `not any(values)`, which in
        // Python treats every falsy scalar (None, 0, 0.0, False, "", [],
        // {}) as "empty". If we don't mirror that here, the gateway's
        // `compute_bundle_config_hash` can diverge from the config
        // service's hash for data like `{"flag": 0}` — the config
        // service would strip the field and hash `{}`, the gateway would
        // keep it and hash `{"flag": 0}`, and every worker in that
        // bundle would sit in `pending_workers` forever because its
        // advertised `bundle_config_hash` never matches the gateway's
        // expected hash. See `canonicalize_adapter_options` for the
        // exact predicate.
        let adapter_options = profile
            .adapter_options
            .clone()
            .and_then(canonicalize_adapter_options);

        Self {
            adapter_path: profile.adapter_path.clone(),
            max_batch_tokens: profile.max_batch_tokens,
            compute_precision: profile.compute_precision.clone(),
            adapter_options,
        }
    }
}

/// Mirror of Python's `not any(adapter_opts.values())` falsy check.
///
/// Returns `None` if `opts` is an object whose values are ALL Python-falsy
/// (null, `false`, `0`, `0.0`, `""`, empty array, empty object). Otherwise
/// returns `Some(opts)` unchanged.
fn canonicalize_adapter_options(opts: serde_json::Value) -> Option<serde_json::Value> {
    if let serde_json::Value::Object(map) = opts {
        let compact: serde_json::Map<String, serde_json::Value> = map
            .into_iter()
            .filter(
                |(_, value)| !matches!(value, serde_json::Value::Object(inner) if inner.is_empty()),
            )
            .collect();
        let all_falsy = compact.values().all(|v| match v {
            serde_json::Value::Null => true,
            serde_json::Value::Bool(b) => !*b,
            serde_json::Value::Number(n) => {
                n.as_f64().map(|f| f == 0.0).unwrap_or(false)
                    || n.as_i64().map(|i| i == 0).unwrap_or(false)
                    || n.as_u64().map(|u| u == 0).unwrap_or(false)
            }
            serde_json::Value::String(s) => s.is_empty(),
            serde_json::Value::Array(a) => a.is_empty(),
            serde_json::Value::Object(o) => o.is_empty(),
        });
        if all_falsy {
            return None;
        }
        return Some(serde_json::Value::Object(compact));
    }
    Some(opts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonical_profile_basic() {
        let profile = ProfileConfig {
            adapter_path: Some("module:Adapter".into()),
            max_batch_tokens: Some(4096),
            compute_precision: Some("float16".into()),
            adapter_options: None,
            extends: None,
        };
        let canonical = CanonicalProfile::from_profile(&profile);
        assert_eq!(canonical.adapter_path, Some("module:Adapter".into()));
        assert_eq!(canonical.max_batch_tokens, Some(4096));
        assert_eq!(canonical.compute_precision, Some("float16".into()));
        assert!(canonical.adapter_options.is_none());
    }

    #[test]
    fn test_canonical_profile_strips_null_only_options() {
        let profile = ProfileConfig {
            adapter_path: Some("mod:A".into()),
            max_batch_tokens: None,
            compute_precision: None,
            adapter_options: Some(serde_json::json!({"key": null})),
            extends: None,
        };
        let canonical = CanonicalProfile::from_profile(&profile);
        assert!(canonical.adapter_options.is_none());
    }

    #[test]
    fn test_canonical_profile_strips_false_only_options() {
        let profile = ProfileConfig {
            adapter_path: Some("mod:A".into()),
            max_batch_tokens: None,
            compute_precision: None,
            adapter_options: Some(serde_json::json!({"enabled": false})),
            extends: None,
        };
        let canonical = CanonicalProfile::from_profile(&profile);
        assert!(canonical.adapter_options.is_none());
    }

    #[test]
    fn test_canonical_profile_keeps_meaningful_options() {
        let opts = serde_json::json!({"batch_size": 32, "key": null});
        let profile = ProfileConfig {
            adapter_path: Some("mod:A".into()),
            max_batch_tokens: None,
            compute_precision: None,
            adapter_options: Some(opts.clone()),
            extends: None,
        };
        let canonical = CanonicalProfile::from_profile(&profile);
        assert_eq!(canonical.adapter_options, Some(opts));
    }

    #[test]
    fn test_canonical_profile_drops_empty_nested_option_maps() {
        let profile = ProfileConfig {
            adapter_path: Some("mod:A".into()),
            max_batch_tokens: None,
            compute_precision: None,
            adapter_options: Some(serde_json::json!({
                "loadtime": {},
                "runtime": {"pooling": "cls"}
            })),
            extends: None,
        };
        let canonical = CanonicalProfile::from_profile(&profile);
        assert_eq!(
            canonical.adapter_options,
            Some(serde_json::json!({"runtime": {"pooling": "cls"}}))
        );
    }

    #[test]
    fn test_canonical_profile_conformance_vectors() {
        // Generic flat canonicalizer: strip an options map to None when every
        // value is falsy (0, 0.0, "", [], {}, None, False) and drop empty nested
        // maps. The config service must match or `compute_bundle_config_hash`
        // diverges and every worker in the affected bundle sits in
        // `pending_workers` forever. These vectors are shared with the sie_config
        // and sie_server Python suites — see
        // `conformance/bundle_config_hash/canonical_profile_falsy_vectors.json`
        // — so all three implementations stay in lockstep (#1542). The
        // `transform` cases pin the loadtime/runtime compaction that real
        // profiles use, which the worker's structured canonicalizer also honors.
        let raw = include_str!(
            "../../../../conformance/bundle_config_hash/canonical_profile_falsy_vectors.json"
        );
        let vectors: serde_json::Value = serde_json::from_str(raw).expect("valid conformance JSON");

        let canonicalize = |opts: &serde_json::Value| {
            CanonicalProfile::from_profile(&ProfileConfig {
                adapter_path: Some("mod:A".into()),
                max_batch_tokens: None,
                compute_precision: None,
                adapter_options: Some(opts.clone()),
                extends: None,
            })
            .adapter_options
        };

        for case in vectors["strip_to_none"]
            .as_array()
            .expect("strip_to_none array")
        {
            let opts = &case["adapter_options"];
            let name = case["name"].as_str().unwrap_or("?");
            assert!(
                canonicalize(opts).is_none(),
                "case {name}: expected falsy-only options {opts:?} to strip to None"
            );
        }

        for case in vectors["keep_unchanged"]
            .as_array()
            .expect("keep_unchanged array")
        {
            let opts = &case["adapter_options"];
            let name = case["name"].as_str().unwrap_or("?");
            let got = canonicalize(opts);
            assert_eq!(
                got.as_ref(),
                Some(opts),
                "case {name}: expected meaningful options to be kept unchanged"
            );
        }

        for case in vectors["transform"].as_array().expect("transform array") {
            let opts = &case["adapter_options"];
            let name = case["name"].as_str().unwrap_or("?");
            let expected = &case["expected"];
            let got = canonicalize(opts);
            if expected.is_null() {
                assert!(
                    got.is_none(),
                    "case {name}: expected {opts:?} to canonicalize to None"
                );
            } else {
                assert_eq!(
                    got.as_ref(),
                    Some(expected),
                    "case {name}: unexpected canonicalization of {opts:?}"
                );
            }
        }
    }

    #[test]
    fn test_canonical_profile_keeps_nonzero_numbers() {
        let profile = ProfileConfig {
            adapter_path: Some("mod:A".into()),
            max_batch_tokens: None,
            compute_precision: None,
            adapter_options: Some(serde_json::json!({"x": 1})),
            extends: None,
        };
        let canonical = CanonicalProfile::from_profile(&profile);
        assert!(canonical.adapter_options.is_some());
    }

    #[test]
    fn test_canonical_profile_equality() {
        let p1 = ProfileConfig {
            adapter_path: Some("mod:A".into()),
            max_batch_tokens: Some(4096),
            compute_precision: None,
            adapter_options: None,
            extends: None,
        };
        let p2 = p1.clone();
        assert_eq!(
            CanonicalProfile::from_profile(&p1),
            CanonicalProfile::from_profile(&p2)
        );
    }

    #[test]
    fn test_model_config_yaml_deserialization() {
        let yaml = r#"
name: BAAI/bge-m3
profiles:
  default:
    adapter_path: "module:Adapter"
    max_batch_tokens: 4096
pool: customer-a
"#;
        let config: ModelConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.name, "BAAI/bge-m3");
        assert_eq!(config.pool, Some("customer-a".into()));
        assert_eq!(config.profiles.len(), 1);
        assert_eq!(
            config.profiles["default"].adapter_path,
            Some("module:Adapter".into())
        );
    }

    #[test]
    fn test_model_config_sie_id_alias() {
        let yaml = r#"
sie_id: my/model
profiles: {}
"#;
        let config: ModelConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.name, "my/model");
    }

    #[test]
    fn test_model_config_json_roundtrip() {
        let config = ModelConfig {
            name: "test/model".into(),
            hf_revision: Some("0123456789abcdef0123456789abcdef01234567".into()),
            adapter_module: Some("mod".into()),
            default_bundle: None,
            pool: Some("customer-a".into()),
            profiles: HashMap::new(),
            inputs: None,
            max_sequence_length: None,
            tasks: None,
        };
        let json = serde_json::to_string(&config).unwrap();
        let back: ModelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, "test/model");
        assert_eq!(back.hf_revision, config.hf_revision);
        assert_eq!(back.adapter_module, Some("mod".into()));
        assert_eq!(back.pool, Some("customer-a".into()));
    }

    #[test]
    fn test_model_info_extras_defaults_dense_when_tasks_absent() {
        let raw: serde_yaml::Value = serde_yaml::from_str(
            r#"
name: example/model
inputs:
  text: true
"#,
        )
        .unwrap();

        let extras = ModelInfoExtras::from_yaml_raw(&raw);
        assert_eq!(extras.outputs, vec!["dense"]);
    }

    #[test]
    fn test_model_info_extras_does_not_invent_dense_for_non_encode_tasks() {
        // A score task should surface ``outputs=["score"]`` (matching the
        // Python ``ModelConfig.outputs`` property in
        // ``packages/sie_server/src/sie_server/config/model.py``), *not* a
        // synthesized ``"dense"``. ``dims`` stays empty because score has no
        // dimension to advertise.
        let raw: serde_yaml::Value = serde_yaml::from_str(
            r#"
name: example/reranker
inputs:
  text: true
tasks:
  score: {}
"#,
        )
        .unwrap();

        let extras = ModelInfoExtras::from_yaml_raw(&raw);
        assert_eq!(extras.outputs, vec!["score"]);
        assert!(extras.dims.is_empty());
    }

    #[test]
    fn test_model_info_extras_lora_adapters_union_across_profiles() {
        // The model-level ``lora_adapters`` summary stays as the union
        // across profiles (back-compat for ``/v1/models`` consumers that
        // listed advertised adapters without caring about profile
        // scoping). The precise per-profile breakdown lives alongside
        // in ``profile_lora_adapters`` and is what the validation gate
        // checks (see ``test_per_profile_lora_capabilities_isolates_profiles``).
        let raw: serde_yaml::Value = serde_yaml::from_str(
            r#"
name: acme/qwen-lora
inputs:
  text: true
tasks:
  generate: {}
profiles:
  default:
    adapter_options:
      loadtime:
        lora_paths:
          acme-support: acme/support-lora
          acme-legal: acme/legal-lora
  a100:
    adapter_options:
      loadtime:
        lora_paths:
          acme-support: acme/support-lora
"#,
        )
        .unwrap();
        let mut names = ModelInfoExtras::from_yaml_raw(&raw)
            .lora_adapters
            .expect("lora_adapters present");
        names.sort();
        assert_eq!(
            names,
            vec!["acme-legal".to_string(), "acme-support".to_string()]
        );
    }

    #[test]
    fn test_per_profile_lora_capabilities_isolates_profiles() {
        // M10 regression: ``profile_lora_adapters`` must scope adapters
        // per declared profile, never collapse to the union. Profile A
        // carries [a1, a2]; profile B carries [b1]; the per-profile map
        // returns ONLY that profile's adapters on lookup.
        let raw: serde_yaml::Value = serde_yaml::from_str(
            r#"
name: acme/multi-profile-lora
inputs:
  text: true
tasks:
  generate: {}
profiles:
  default:
    adapter_options:
      loadtime:
        lora_paths:
          a1: acme/a1
          a2: acme/a2
  a100:
    adapter_options:
      loadtime:
        lora_paths:
          b1: acme/b1
"#,
        )
        .unwrap();
        let extras = ModelInfoExtras::from_yaml_raw(&raw);
        let per_profile = extras
            .profile_lora_adapters
            .as_ref()
            .expect("profile_lora_adapters populated");
        let mut default_adapters = per_profile
            .get("default")
            .expect("default profile present")
            .clone();
        default_adapters.sort();
        assert_eq!(
            default_adapters,
            vec!["a1".to_string(), "a2".to_string()],
            "default profile must not include profile A100's adapters"
        );
        let a100_adapters = per_profile
            .get("a100")
            .expect("a100 profile present")
            .clone();
        assert_eq!(
            a100_adapters,
            vec!["b1".to_string()],
            "a100 profile must not include the default profile's adapters"
        );
        // And the union summary is still the deduped flatten.
        let mut union = extras
            .lora_adapters
            .as_ref()
            .expect("union present")
            .clone();
        union.sort();
        assert_eq!(
            union,
            vec!["a1".to_string(), "a2".to_string(), "b1".to_string()]
        );
    }

    #[test]
    fn test_model_info_extras_no_lora_adapters_when_absent() {
        let raw: serde_yaml::Value =
            serde_yaml::from_str("name: m\ninputs:\n  text: true\ntasks:\n  generate: {}\n")
                .unwrap();
        let extras = ModelInfoExtras::from_yaml_raw(&raw);
        assert!(extras.lora_adapters.is_none());
        assert!(extras.profile_lora_adapters.is_none());
    }

    #[test]
    fn test_model_info_extras_tools_capability_parsed() {
        let raw: serde_yaml::Value = serde_yaml::from_str(
            r#"
name: Qwen/Qwen3-4B-Instruct-2507
inputs:
  text: true
tasks:
  generate:
    context_length: 32768
    max_output_tokens: 4096
    capabilities:
      grammar: ["json_schema"]
      tools: true
"#,
        )
        .unwrap();
        let extras = ModelInfoExtras::from_yaml_raw(&raw);
        assert_eq!(extras.tools_supported, Some(true));
    }

    #[test]
    fn test_model_info_extras_code_sql_guard_capabilities_parsed() {
        // Qwen3-4B-Instruct-2507 advertises code + sql (and tools); it is
        // not a guard model. Mirrors
        // ``packages/sie_server/models/Qwen__Qwen3-4B-Instruct-2507.yaml``.
        let raw: serde_yaml::Value = serde_yaml::from_str(
            r#"
name: Qwen/Qwen3-4B-Instruct-2507
inputs:
  text: true
tasks:
  generate:
    context_length: 32768
    max_output_tokens: 4096
    capabilities:
      grammar: ["json_schema", "regex", "ebnf"]
      tools: true
      code: true
      sql: true
"#,
        )
        .unwrap();
        let extras = ModelInfoExtras::from_yaml_raw(&raw);
        assert!(extras.code);
        assert!(extras.sql);
        assert!(!extras.guard);

        // Qwen3.6-27B also advertises code + sql. Mirrors
        // ``packages/sie_server/models/Qwen__Qwen3.6-27B.yaml``.
        let raw: serde_yaml::Value = serde_yaml::from_str(
            r#"
name: Qwen/Qwen3.6-27B
inputs:
  text: true
tasks:
  generate:
    context_length: 32768
    max_output_tokens: 4096
    capabilities:
      grammar: ["json_schema", "regex"]
      tools: true
      code: true
      sql: true
"#,
        )
        .unwrap();
        let extras = ModelInfoExtras::from_yaml_raw(&raw);
        assert!(extras.code);
        assert!(extras.sql);
        assert!(!extras.guard);

        // Granite Guardian 3.0-2b advertises guard only. Mirrors
        // ``packages/sie_server/models/ibm-granite__granite-guardian-3.0-2b.yaml``.
        let raw: serde_yaml::Value = serde_yaml::from_str(
            r#"
name: ibm-granite/granite-guardian-3.0-2b
inputs:
  text: true
tasks:
  generate:
    context_length: 8192
    max_output_tokens: 512
    capabilities:
      grammar: []
      tools: false
      guard: true
"#,
        )
        .unwrap();
        let extras = ModelInfoExtras::from_yaml_raw(&raw);
        assert!(extras.guard);
        assert!(!extras.code);
        assert!(!extras.sql);
    }

    #[test]
    fn test_model_info_extras_code_sql_guard_default_false_when_absent() {
        // No ``capabilities`` block at all — all three default to false.
        let raw: serde_yaml::Value = serde_yaml::from_str(
            r#"
name: m
inputs:
  text: true
tasks:
  generate:
    context_length: 1024
    max_output_tokens: 512
"#,
        )
        .unwrap();
        let extras = ModelInfoExtras::from_yaml_raw(&raw);
        assert!(!extras.code);
        assert!(!extras.sql);
        assert!(!extras.guard);
    }

    #[test]
    fn test_to_model_info_value_surfaces_code_sql_guard_flags() {
        // The three flags must surface inside the same ``capabilities``
        // JSON object as ``grammar``/``tools``, using snake_case wire keys.
        let raw: serde_yaml::Value = serde_yaml::from_str(
            r#"
name: Qwen/Qwen3-4B-Instruct-2507
inputs:
  text: true
tasks:
  generate:
    context_length: 32768
    max_output_tokens: 4096
    capabilities:
      grammar: ["json_schema"]
      tools: true
      code: true
      sql: true
"#,
        )
        .unwrap();
        let info_extras = ModelInfoExtras::from_yaml_raw(&raw);
        let entry = ModelEntry {
            name: "Qwen/Qwen3-4B-Instruct-2507".to_string(),
            pool: None,
            bundles: Vec::new(),
            adapter_modules: HashSet::new(),
            profile_names: HashSet::new(),
            profile_configs: HashMap::new(),
            info_extras,
        };
        let body = entry.to_model_info_value(false);
        let caps = body
            .get("capabilities")
            .and_then(|c| c.as_object())
            .expect("capabilities block");
        assert_eq!(caps.get("code"), Some(&json!(true)));
        assert_eq!(caps.get("sql"), Some(&json!(true)));
        assert_eq!(caps.get("guard"), Some(&json!(false)));
    }

    #[test]
    fn test_model_info_extras_tools_capability_defaults_to_false() {
        let raw: serde_yaml::Value = serde_yaml::from_str(
            r#"
name: m
inputs:
  text: true
tasks:
  generate:
    context_length: 1024
    max_output_tokens: 512
"#,
        )
        .unwrap();
        let extras = ModelInfoExtras::from_yaml_raw(&raw);
        assert_eq!(extras.tools_supported, Some(false));
    }

    #[test]
    fn test_model_info_extras_generate_task_surfaces_tokens_output() {
        // A generate task surfaces ``["tokens"]`` so
        // ``GET /v1/models`` advertises the capability accurately. Without
        // this, generate-only models report ``outputs=[]`` and downstream
        // clients can't tell what the model produces.
        let raw: serde_yaml::Value = serde_yaml::from_str(
            r#"
name: Qwen/Qwen3-4B-Instruct
inputs:
  text: true
tasks:
  generate:
    context_length: 32768
    max_output_tokens: 4096
"#,
        )
        .unwrap();

        let extras = ModelInfoExtras::from_yaml_raw(&raw);
        assert_eq!(extras.outputs, vec!["tokens"]);
    }

    #[test]
    fn test_supports_vision_generation_requires_image_and_generate() {
        // Vision-capable generation model (image input + generate task).
        let vlm = ModelInfoExtras::from_yaml_raw(
            &serde_yaml::from_str(
                "name: m\ninputs:\n  text: true\n  image: true\ntasks:\n  generate: {}\n",
            )
            .unwrap(),
        );
        assert!(vlm.supports_vision_generation());

        // Encode-only image model (CLIP/SigLIP-style: image input, no generate
        // task) must NOT pass the vision-generation gate.
        let encode_only = ModelInfoExtras::from_yaml_raw(
            &serde_yaml::from_str(
                "name: m\ninputs:\n  text: true\n  image: true\ntasks:\n  encode: {}\n",
            )
            .unwrap(),
        );
        assert!(!encode_only.supports_vision_generation());

        // Text-only generation model: generates but no image input.
        let text_gen = ModelInfoExtras::from_yaml_raw(
            &serde_yaml::from_str("name: m\ninputs:\n  text: true\ntasks:\n  generate: {}\n")
                .unwrap(),
        );
        assert!(!text_gen.supports_vision_generation());
    }

    #[test]
    fn test_lora_adapters_for_profile_returns_scoped_list() {
        // ``ModelEntry::lora_adapters_for_profile`` is the validation
        // primitive — must hand back ONLY the requested profile's
        // adapters even when sibling profiles declare different ones.
        // This is the seam the proxy.rs lora_adapter gate consumes.
        let raw: serde_yaml::Value = serde_yaml::from_str(
            r#"
name: acme/multi-profile-lora
inputs:
  text: true
tasks:
  generate: {}
profiles:
  default:
    adapter_options:
      loadtime:
        lora_paths:
          a1: acme/a1
  a100:
    adapter_options:
      loadtime:
        lora_paths:
          b1: acme/b1
"#,
        )
        .unwrap();
        let info_extras = ModelInfoExtras::from_yaml_raw(&raw);
        let entry = ModelEntry {
            name: "acme/multi-profile-lora".to_string(),
            pool: None,
            bundles: Vec::new(),
            adapter_modules: HashSet::new(),
            profile_names: ["default".to_string(), "a100".to_string()]
                .iter()
                .cloned()
                .collect(),
            profile_configs: HashMap::new(),
            info_extras,
        };
        assert_eq!(
            entry.lora_adapters_for_profile("default"),
            Some(&vec!["a1".to_string()]),
        );
        assert_eq!(
            entry.lora_adapters_for_profile("a100"),
            Some(&vec!["b1".to_string()]),
        );
        // Profiles that exist but declare no adapters (or that don't
        // exist at all) both yield ``None`` here — the gate uses that
        // to reject any ``lora_adapter`` request against them.
        assert!(entry.lora_adapters_for_profile("missing").is_none());
    }

    #[test]
    fn test_to_model_info_value_advertises_per_profile_lora_breakdown() {
        // ``/v1/models`` must surface both the union (back-compat) and
        // the per-profile map (G-M10). Vanilla OpenAI clients keep
        // reading the union; consumers that need precise routing scope
        // read the per-profile breakdown without reverse-engineering it.
        let raw: serde_yaml::Value = serde_yaml::from_str(
            r#"
name: acme/multi-profile-lora
inputs:
  text: true
tasks:
  generate: {}
profiles:
  default:
    adapter_options:
      loadtime:
        lora_paths:
          a1: acme/a1
  a100:
    adapter_options:
      loadtime:
        lora_paths:
          b1: acme/b1
"#,
        )
        .unwrap();
        let info_extras = ModelInfoExtras::from_yaml_raw(&raw);
        let entry = ModelEntry {
            name: "acme/multi-profile-lora".to_string(),
            pool: None,
            bundles: Vec::new(),
            adapter_modules: HashSet::new(),
            profile_names: ["default".to_string(), "a100".to_string()]
                .iter()
                .cloned()
                .collect(),
            profile_configs: HashMap::new(),
            info_extras,
        };
        let body = entry.to_model_info_value(false);
        let caps = body
            .get("capabilities")
            .and_then(|c| c.as_object())
            .expect("capabilities block");
        // Union stays present for back-compat.
        let mut union: Vec<String> = caps
            .get("lora_adapters")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .expect("lora_adapters union present");
        union.sort();
        assert_eq!(union, vec!["a1".to_string(), "b1".to_string()]);
        // Per-profile breakdown carries the precise scope.
        let per_profile = caps
            .get("profile_lora_adapters")
            .and_then(|v| v.as_object())
            .expect("profile_lora_adapters present");
        let default_adapters: Vec<String> = per_profile
            .get("default")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .expect("default profile entry");
        assert_eq!(default_adapters, vec!["a1".to_string()]);
        let a100_adapters: Vec<String> = per_profile
            .get("a100")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .expect("a100 profile entry");
        assert_eq!(a100_adapters, vec!["b1".to_string()]);
    }
}
