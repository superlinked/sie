//! Per-model tokenizer registry, lazily populated from
//! `EnsureModelReady` handshake descriptors.
//!
//! # How tokenisers get into the registry
//!
//! The registry is **adapter-driven**: the
//! Python (or future native) adapter declares its tokeniser path,
//! content hash, and max-seq-len on the first
//! `EnsureModelReady(model_id)` for that model, and the dispatcher
//! calls [`TokenizerRegistry::register_from_descriptor`] to ingest
//! the entry. There is no sidecar-local model list that gates this path.
//!
//! # Concurrency
//!
//! Entries are stored behind an `RwLock` so the dispatcher's
//! per-batch lookup ([`TokenizerRegistry::get`]) is wait-free under
//! contention from the rare ingest path. `TokenizerEntry` is `Clone`
//! (the underlying `Tokenizer` is wrapped in [`Arc`]), so the read
//! path returns an owned snapshot and the lock is released before
//! tokenisation runs.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use tokenizers::Tokenizer;
use tracing::{info, warn};

use super::{load_no_pad_tokenizer, tokenize_no_pad, tokenizer_content_hash, RaggedTokens};
use crate::ipc_types::ModelDescriptor;

/// Fallback max sequence length when an entry doesn't carry one.
/// Matches the historical `max_seq_len` default used across adapters
/// so operators see consistent behaviour.
pub const DEFAULT_MAX_SEQ_LEN: usize = 512;

/// Loaded per-model tokenizer.
#[derive(Clone)]
pub struct TokenizerEntry {
    /// Underlying HF fast-tokenizer, configured with no padding +
    /// truncate-to-`max_seq_len`. Shared across worker threads via
    /// [`Arc`]; `tokenizers::Tokenizer::encode_batch` is itself
    /// thread-safe (immutable borrow).
    tokenizer: Arc<Tokenizer>,
    /// Per-model truncation bound. Already baked into `tokenizer`'s
    /// truncation config; kept here for logging / metrics.
    max_seq_len: usize,
    /// Stable content hash of the `tokenizer.json` bytes used to
    /// construct this entry. Emitted over the wire so Python can
    /// verify it matches the tokenizer its adapter loaded.
    tokenizer_id: String,
    /// Model-default `query_template`, sourced from the adapter's
    /// `EnsureModelReady` descriptor. Selected when an item arrives
    /// with `is_query=true` and its `options.query_template` is
    /// absent. `None` keeps today's behaviour (no template
    /// application on the Rust side for this model).
    default_query_template: Option<String>,
    /// Model-default `doc_template` (selected when `is_query=false`).
    /// Same wire contract as [`Self::default_query_template`].
    default_doc_template: Option<String>,
}

impl TokenizerEntry {
    pub fn tokenizer_id(&self) -> &str {
        &self.tokenizer_id
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Model-default `query_template` from the adapter handshake.
    pub fn default_query_template(&self) -> Option<&str> {
        self.default_query_template.as_deref()
    }

    /// Model-default `doc_template` from the adapter handshake.
    pub fn default_doc_template(&self) -> Option<&str> {
        self.default_doc_template.as_deref()
    }

    /// Tokenise a batch of texts with this entry's tokenizer. Ragged
    /// output (no padding), truncated at `max_seq_len`. See
    /// [`tokenize_no_pad`].
    pub fn tokenize(&self, texts: &[&str]) -> Result<RaggedTokens, String> {
        tokenize_no_pad(self.tokenizer.as_ref(), texts)
    }
}

impl std::fmt::Debug for TokenizerEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenizerEntry")
            .field("max_seq_len", &self.max_seq_len)
            .field("tokenizer_id", &self.tokenizer_id)
            .field("default_query_template", &self.default_query_template)
            .field("default_doc_template", &self.default_doc_template)
            .finish()
    }
}

/// Per-model tokenizer lookup. Always exists — empty registry
/// just means "no model has registered a tokeniser yet" and every
/// [`TokenizerRegistry::get`] returns `None`. Wrapped in an [`Arc`]
/// so the dispatcher and any future callers share a single mutable
/// table.
#[derive(Debug)]
pub struct TokenizerRegistry {
    entries: RwLock<HashMap<String, TokenizerEntry>>,
}

impl TokenizerRegistry {
    /// Build an empty registry. Tokenisers are added later via
    /// [`Self::register_from_descriptor`] during the IPC handshake.
    pub fn empty() -> Arc<Self> {
        Arc::new(Self {
            entries: RwLock::new(HashMap::new()),
        })
    }

    /// Look up the tokenizer for a model id. Returns an owned (cloned)
    /// entry so the caller can drop the read lock before running
    /// `encode_batch`. `Arc<Tokenizer>` makes the clone cheap.
    /// Absent → caller must fall back to the Python tokenise path
    /// (no `prepared_tokens` on the wire).
    pub fn get(&self, model_id: &str) -> Option<TokenizerEntry> {
        self.entries
            .read()
            .expect("tokenizer registry RwLock poisoned")
            .get(model_id)
            .cloned()
    }

    /// True iff a tokeniser is registered for `model_id`.
    pub fn contains(&self, model_id: &str) -> bool {
        self.entries
            .read()
            .expect("tokenizer registry RwLock poisoned")
            .contains_key(model_id)
    }

    pub fn len(&self) -> usize {
        self.entries
            .read()
            .expect("tokenizer registry RwLock poisoned")
            .len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Sorted list of model ids — used for startup log output and
    /// operator introspection.
    pub fn model_ids(&self) -> Vec<String> {
        let guard = self
            .entries
            .read()
            .expect("tokenizer registry RwLock poisoned");
        let mut ids: Vec<String> = guard.keys().cloned().collect();
        ids.sort_unstable();
        ids
    }

    /// Ingest a tokeniser declared by an adapter on the
    /// `EnsureModelReady` handshake.
    ///
    /// Returns:
    ///
    /// * `Ok(true)`  — entry was newly inserted (first time we've
    ///   seen this `model_id` with a tokenizer_path).
    /// * `Ok(false)` — entry already present, descriptor has no
    ///   `tokenizer_path`, or the descriptor's tokenizer identity and
    ///   effective configuration match the cached entry (idempotent
    ///   re-handshake).
    /// * `Err(_)`    — descriptor named a path that fails to load
    ///   (file missing, bad JSON, ...). The caller should log and
    ///   continue: the model just falls back to Python tokenisation.
    ///
    /// If the existing entry differs in `tokenizer_id`, effective
    /// `max_seq_len`, or default templates, we replace it — the adapter is the
    /// source of truth, and a hot-reload of tokeniser configuration must be
    /// reflected here. Missing or zero descriptor caps normalize to
    /// [`DEFAULT_MAX_SEQ_LEN`] before this identity comparison.
    pub fn register_from_descriptor(
        &self,
        model_id: &str,
        descriptor: &ModelDescriptor,
    ) -> Result<bool, String> {
        let Some(path_str) = descriptor.tokenizer_path.as_deref() else {
            return Ok(false);
        };
        if path_str.is_empty() {
            return Ok(false);
        }

        let max_seq_len = descriptor_max_seq_len(descriptor.max_seq_len);

        // Idempotent: if we already loaded a tokeniser for this model
        // and the declared id + effective cap + templates match, do nothing. Saves the
        // cost of re-reading + re-hashing on every batch's
        // ensure_ready call. The cap and templates are part of the equality
        // check so YAML-driven edits hot-reload on the next handshake without
        // restarting the sidecar.
        if let Some(existing) = self.get(model_id) {
            let config_matches = existing.max_seq_len == max_seq_len
                && existing.default_query_template.as_deref()
                    == descriptor.default_query_template.as_deref()
                && existing.default_doc_template.as_deref()
                    == descriptor.default_doc_template.as_deref();
            if let Some(decl_id) = descriptor.tokenizer_id.as_deref() {
                if existing.tokenizer_id() == decl_id && config_matches {
                    return Ok(false);
                }
            } else if config_matches {
                return Ok(false);
            }
        }

        let path = PathBuf::from(path_str);
        let entry = load_entry_with_templates(
            &path,
            max_seq_len,
            descriptor.default_query_template.clone(),
            descriptor.default_doc_template.clone(),
        )?;

        // Defence in depth: if the adapter declared a `tokenizer_id`
        // and our locally-computed hash disagrees, refuse to register
        // rather than enabling a fast path that the Python side will
        // immediately reject. The mismatch is loud at `warn` so an
        // operator notices the path/version drift.
        if let Some(decl_id) = descriptor.tokenizer_id.as_deref() {
            if entry.tokenizer_id != decl_id {
                warn!(
                    model = %model_id,
                    declared_id = %decl_id,
                    loaded_id = %entry.tokenizer_id,
                    path = %path.display(),
                    "tokenize: descriptor tokenizer_id disagrees with loaded file; \
                     skipping registration — Python will tokenise this model"
                );
                return Ok(false);
            }
        }

        let mut guard = self
            .entries
            .write()
            .expect("tokenizer registry RwLock poisoned");
        let newly_inserted = !guard.contains_key(model_id);
        info!(
            model = %model_id,
            max_seq_len = entry.max_seq_len,
            tokenizer_id = %entry.tokenizer_id,
            path = %path.display(),
            replaced = !newly_inserted,
            "tokenize: registered tokeniser from EnsureModelReady descriptor"
        );
        guard.insert(model_id.to_string(), entry);
        Ok(newly_inserted)
    }
}

fn descriptor_max_seq_len(max_seq_len: Option<u32>) -> usize {
    max_seq_len
        .map(|value| value as usize)
        .filter(|&value| value > 0)
        .unwrap_or(DEFAULT_MAX_SEQ_LEN)
}

impl Default for TokenizerRegistry {
    fn default() -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
        }
    }
}

fn load_entry_with_templates(
    path: &Path,
    max_seq_len: usize,
    default_query_template: Option<String>,
    default_doc_template: Option<String>,
) -> Result<TokenizerEntry, String> {
    if !path.is_file() {
        return Err(format!("tokenizer.json not found: {}", path.display()));
    }
    // Hash BEFORE applying our IPC-specific padding/truncation overrides
    // so the id identifies the *adapter's* tokeniser, not our runtime
    // configuration. See `tokenizer_content_hash` for the rationale.
    let hash_source =
        Tokenizer::from_file(path).map_err(|e| format!("load tokenizer for hash: {e}"))?;
    let tokenizer_id = tokenizer_content_hash(&hash_source);
    drop(hash_source);

    let tokenizer = load_no_pad_tokenizer(path, max_seq_len)?;
    Ok(TokenizerEntry {
        tokenizer: Arc::new(tokenizer),
        max_seq_len,
        tokenizer_id,
        default_query_template,
        default_doc_template,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    /// Minimal `tokenizer.json` blob for a WordLevel tokenizer. The
    /// byte-identity integration test uses a real HF model; here we
    /// just need a JSON the `tokenizers` crate will accept so the
    /// registry-loading path is exercised end-to-end without a
    /// network fetch.
    const TINY_TOKENIZER_JSON: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": { "type": "Whitespace" },
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": { "[UNK]": 0, "hello": 1, "world": 2 },
    "unk_token": "[UNK]"
  }
}"#;

    fn write_tiny_tokenizer(dir: &TempDir) -> PathBuf {
        let path = dir.path().join("tokenizer.json");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(TINY_TOKENIZER_JSON.as_bytes()).unwrap();
        path
    }

    #[test]
    fn empty_registry_is_zero_cost() {
        let reg = TokenizerRegistry::empty();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
        assert!(reg.get("any-model").is_none());
        assert!(!reg.contains("any-model"));
    }

    // -- Handshake path ----------------------------------------------------

    #[test]
    fn register_from_descriptor_loads_tokenizer_and_returns_true() {
        let dir = TempDir::new().unwrap();
        let path = write_tiny_tokenizer(&dir);
        let reg = TokenizerRegistry::empty();

        let descriptor = ModelDescriptor {
            tokenizer_path: Some(path.display().to_string()),
            tokenizer_id: None,
            max_seq_len: Some(64),
            output_types: vec!["dense".into()],
            supports_run_batch: true,
            ..Default::default()
        };

        let inserted = reg
            .register_from_descriptor("test-model", &descriptor)
            .expect("descriptor load should succeed");
        assert!(inserted);
        assert_eq!(reg.len(), 1);

        let entry = reg.get("test-model").unwrap();
        assert_eq!(entry.max_seq_len(), 64);
        assert_eq!(entry.tokenizer_id().len(), 32);

        // Tokenisation works through the descriptor-loaded entry.
        let out = entry.tokenize(&["hello world"]).unwrap();
        assert_eq!(out.input_ids[0], vec![1, 2]);
    }

    #[test]
    fn register_from_descriptor_is_idempotent_on_matching_hash() {
        let dir = TempDir::new().unwrap();
        let path = write_tiny_tokenizer(&dir);
        let reg = TokenizerRegistry::empty();

        let mut descriptor = ModelDescriptor {
            tokenizer_path: Some(path.display().to_string()),
            tokenizer_id: None,
            max_seq_len: Some(64),
            ..Default::default()
        };
        // First call: insert. Capture the loaded tokenizer_id.
        assert!(reg
            .register_from_descriptor("test-model", &descriptor)
            .unwrap());
        let loaded_id = reg.get("test-model").unwrap().tokenizer_id().to_string();

        // Second call with declared_id == loaded_id: no-op.
        descriptor.tokenizer_id = Some(loaded_id);
        assert!(!reg
            .register_from_descriptor("test-model", &descriptor)
            .unwrap());
    }

    #[test]
    fn register_from_descriptor_replaces_entry_when_max_seq_len_changes() {
        let dir = TempDir::new().unwrap();
        let path = write_tiny_tokenizer(&dir);
        let reg = TokenizerRegistry::empty();

        let mut descriptor = ModelDescriptor {
            tokenizer_path: Some(path.display().to_string()),
            tokenizer_id: None,
            max_seq_len: Some(4),
            ..Default::default()
        };
        assert!(reg
            .register_from_descriptor("test-model", &descriptor)
            .unwrap());
        let loaded_id = reg.get("test-model").unwrap().tokenizer_id().to_string();

        descriptor.tokenizer_id = Some(loaded_id);
        descriptor.max_seq_len = Some(2);
        assert!(!reg
            .register_from_descriptor("test-model", &descriptor)
            .unwrap());

        let entry = reg.get("test-model").unwrap();
        assert_eq!(entry.max_seq_len(), 2);
        let out = entry.tokenize(&["hello world hello world"]).unwrap();
        assert_eq!(out.input_ids[0].len(), 2);
    }

    #[test]
    fn descriptor_max_seq_len_normalizes_missing_and_zero_to_default() {
        assert_eq!(descriptor_max_seq_len(None), DEFAULT_MAX_SEQ_LEN);
        assert_eq!(descriptor_max_seq_len(Some(0)), DEFAULT_MAX_SEQ_LEN);
        assert_eq!(descriptor_max_seq_len(Some(64)), 64);
    }

    #[test]
    fn register_from_descriptor_skips_when_tokenizer_id_mismatches() {
        let dir = TempDir::new().unwrap();
        let path = write_tiny_tokenizer(&dir);
        let reg = TokenizerRegistry::empty();

        // Declared id deliberately wrong → registry refuses to register
        // rather than enabling a fast path the Python side will reject.
        let descriptor = ModelDescriptor {
            tokenizer_path: Some(path.display().to_string()),
            tokenizer_id: Some("deadbeef".repeat(4)),
            max_seq_len: Some(64),
            ..Default::default()
        };
        let inserted = reg
            .register_from_descriptor("test-model", &descriptor)
            .unwrap();
        assert!(!inserted);
        assert!(reg.is_empty());
    }

    #[test]
    fn register_from_descriptor_returns_false_when_path_absent() {
        let reg = TokenizerRegistry::empty();
        let descriptor = ModelDescriptor::default();
        assert!(!reg
            .register_from_descriptor("test-model", &descriptor)
            .unwrap());
        assert!(reg.is_empty());
    }

    #[test]
    fn register_from_descriptor_stores_default_templates() {
        let dir = TempDir::new().unwrap();
        let path = write_tiny_tokenizer(&dir);
        let reg = TokenizerRegistry::empty();
        let descriptor = ModelDescriptor {
            tokenizer_path: Some(path.display().to_string()),
            tokenizer_id: None,
            max_seq_len: Some(64),
            default_query_template: Some("query: {text}".into()),
            default_doc_template: Some("passage: {text}".into()),
            ..Default::default()
        };
        assert!(reg
            .register_from_descriptor("test-model", &descriptor)
            .unwrap());
        let entry = reg.get("test-model").unwrap();
        assert_eq!(entry.default_query_template(), Some("query: {text}"));
        assert_eq!(entry.default_doc_template(), Some("passage: {text}"));
    }

    #[test]
    fn register_from_descriptor_replaces_entry_when_templates_change() {
        // A YAML edit to a template should hot-reload on the next
        // EnsureModelReady — the registry must not stick with the
        // old templates just because the tokenizer hash hasn't moved.
        let dir = TempDir::new().unwrap();
        let path = write_tiny_tokenizer(&dir);
        let reg = TokenizerRegistry::empty();

        let mut descriptor = ModelDescriptor {
            tokenizer_path: Some(path.display().to_string()),
            tokenizer_id: None,
            max_seq_len: Some(64),
            default_query_template: Some("v1-query: {text}".into()),
            default_doc_template: Some("v1-doc: {text}".into()),
            ..Default::default()
        };
        assert!(reg
            .register_from_descriptor("test-model", &descriptor)
            .unwrap());
        let loaded_id = reg.get("test-model").unwrap().tokenizer_id().to_string();

        // Same tokenizer hash + new templates: must replace, not skip.
        descriptor.tokenizer_id = Some(loaded_id);
        descriptor.default_query_template = Some("v2-query: {text}".into());
        // Re-registration returns `false` (entry already existed for
        // this model id) but the stored templates must reflect v2.
        let _ = reg
            .register_from_descriptor("test-model", &descriptor)
            .unwrap();
        let entry = reg.get("test-model").unwrap();
        assert_eq!(entry.default_query_template(), Some("v2-query: {text}"));
        assert_eq!(entry.default_doc_template(), Some("v1-doc: {text}"));
    }

    #[test]
    fn register_from_descriptor_errors_on_missing_file() {
        let reg = TokenizerRegistry::empty();
        let descriptor = ModelDescriptor {
            tokenizer_path: Some("/definitely/not/a/tokenizer.json".into()),
            ..Default::default()
        };
        let err = reg
            .register_from_descriptor("test-model", &descriptor)
            .unwrap_err();
        assert!(err.contains("not found"), "got {err:?}");
        assert!(reg.is_empty());
    }
}
