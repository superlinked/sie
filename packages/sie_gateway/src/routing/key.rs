//! Routing-key resolution + xxh3 hashing.
//!
//! ## Default policy
//!
//! Cache-aware prefix routing is **on by default** with normalized keys
//! (lowercase + collapsed whitespace) and **no salt**. This co-locates
//! requests that share a long system prompt onto one worker so they reuse
//! its SGLang radix-cache KV.
//!
//! ## Knobs (operator opt-in)
//!
//! - **`SIE_ROUTING_PREFIX_MODE`** — one of:
//!   - *unset* / anything not listed below — cache-aware, normalized prefix
//!     (default; lowercases + collapses whitespace before hashing).
//!   - `off` — disables cache-aware routing; falls back to the legacy
//!     whole-window hash of the system message / prompt.
//!   - `byte-preserving` (alias `byte_preserving`, `bytes`) — cache-aware on,
//!     but **skip the lowercase + whitespace-collapse normalization**. The
//!     routing key then matches byte-for-byte what SGLang's radix cache
//!     would see, eliminating false co-location at the cost of routing
//!     whitespace-only-different prompts to different workers. Recommended
//!     when measurements show normalized routing decisions don't correlate
//!     with real cache hits.
//! - **`SIE_ROUTING_PREFIX_BYTES`** — leading-byte window hashed (default 512).
//! - **`SIE_ROUTING_PREFIX_MIN_BYTES`** — below this length the prefix is
//!   too short to pin and the legacy key is used (default 64).
//! - **`SIE_ROUTING_SALT`** — optional tenant/namespace string mixed into
//!   the cache-aware prefix hash. When set, identical system prompts from
//!   different deployments / tenants (each running the gateway with their
//!   own salt) route to different workers, breaking up hot-spots that
//!   otherwise pin all shared-system-prompt traffic to one worker until
//!   saturation fallback kicks in. Default `None` = legacy behaviour.
//!   *Note:* this is a process-wide salt today; per-request tenant salts
//!   (e.g. via an `x-sie-routing-namespace` header) are a future extension
//!   and not wired into the handler chain yet.
//!
//! ## Priority order (first non-empty wins)
//!
//! 1. `routing_key` — caller-supplied affinity hint (exact full-string hash).
//! 2. `prompt_cache_key` — OpenAI-compatible cache key (exact full-string hash).
//! 3. **Cache-aware prefix** — a normalized (or byte-preserving) leading
//!    window of the system message (or prompt), optionally salted; see
//!    [`derive_prefix_key`] and [`PrefixConfig`].
//! 4. Legacy whole-window hash of the system message / prompt (first 512
//!    bytes) — used when the prefix is too short to pin, or `mode=off`.
//! 5. None — caller falls through to round-robin / pool publish.

use std::sync::LazyLock;

use xxhash_rust::xxh3::xxh3_64;

use crate::queue::publisher::{GenerateInput, GenerateParams};

/// Maximum number of prompt bytes hashed when no explicit key is present.
/// Picked to keep hashing trivially cheap on long prompts while still
/// providing enough entropy for prefix-stable distribution.
pub const PROMPT_PREFIX_BYTES: usize = 512;

/// Cache-aware (prefix-hash) routing configuration, read once from the
/// environment. When enabled (default), a normalized leading window of the
/// system/prompt prefix becomes the HRW key so requests sharing a long
/// prefix (shared system prompts, RAG/few-shot preambles) co-locate on one
/// worker and reuse its SGLang radix-cache KV (lower TTFT + prefill cost).
/// Falls back to the legacy whole-window hash when a prefix is too short to
/// be worth pinning, so it is never worse than the prior behaviour.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixConfig {
    /// `false` restores exact legacy behaviour (`SIE_ROUTING_PREFIX_MODE=off`).
    pub enabled: bool,
    /// Leading bytes of the prefix that are hashed.
    pub prefix_bytes: usize,
    /// Below this length the prefix is not worth pinning; fall back to the
    /// legacy key. Compared against the *normalized* length in normalized
    /// mode and against the *raw* byte length in byte-preserving mode.
    pub min_bytes: usize,
    /// When `true`, skip the lowercase + whitespace-collapse normalization
    /// before hashing. The routing key then matches byte-for-byte what
    /// SGLang's radix cache would see, eliminating false co-location at the
    /// cost of routing whitespace-only-different prompts to different
    /// workers. Default `false` (current behaviour).
    pub byte_preserving: bool,
    /// Optional process-wide tenant / namespace salt. When `Some`, the salt
    /// bytes are mixed into the hash before the prefix bytes, so identical
    /// system prompts under different salts produce different routing keys.
    /// Default `None`.
    pub salt: Option<String>,
}

impl Default for PrefixConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            prefix_bytes: PROMPT_PREFIX_BYTES,
            min_bytes: 64,
            byte_preserving: false,
            salt: None,
        }
    }
}

impl PrefixConfig {
    /// Read `SIE_ROUTING_PREFIX_MODE` / `_BYTES` / `_MIN_BYTES` / `_SALT`
    /// with the documented defaults and bounds (`prefix_bytes` clamped to
    /// `1..=4096`). `SIE_ROUTING_PREFIX_MODE` accepts:
    ///
    /// - *unset* / anything else — cache-aware on, normalized (default).
    /// - `off` — cache-aware off (legacy whole-window hash).
    /// - `byte-preserving` / `byte_preserving` / `bytes` — cache-aware on,
    ///   normalization disabled.
    pub fn from_env() -> Self {
        let d = Self::default();
        let raw_mode = std::env::var("SIE_ROUTING_PREFIX_MODE").ok();
        let (enabled, byte_preserving) = match raw_mode.as_deref() {
            Some(v) if v.eq_ignore_ascii_case("off") => (false, false),
            Some(v)
                if v.eq_ignore_ascii_case("byte-preserving")
                    || v.eq_ignore_ascii_case("byte_preserving")
                    || v.eq_ignore_ascii_case("bytes") =>
            {
                (true, true)
            }
            _ => (d.enabled, d.byte_preserving),
        };
        let prefix_bytes = std::env::var("SIE_ROUTING_PREFIX_BYTES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(d.prefix_bytes)
            .clamp(1, 4096);
        let min_bytes = std::env::var("SIE_ROUTING_PREFIX_MIN_BYTES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(d.min_bytes);
        let salt = std::env::var("SIE_ROUTING_SALT")
            .ok()
            .filter(|s| !s.is_empty());
        Self {
            enabled,
            prefix_bytes,
            min_bytes,
            byte_preserving,
            salt,
        }
    }
}

static PREFIX_CONFIG: LazyLock<PrefixConfig> = LazyLock::new(PrefixConfig::from_env);

/// Which input the routing key was sourced from. Surfaced as a metric label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeySource {
    RoutingKey,
    PromptCacheKey,
    /// Cache-aware: a normalized leading window of the system/prompt prefix
    /// (see [`derive_prefix_key`]). Distinct from [`KeySource::PromptPrefix`]
    /// / [`KeySource::SystemMessage`], which hash the raw whole window.
    PromptPrefixCacheAware,
    PromptPrefix,
    SystemMessage,
    None,
}

impl KeySource {
    /// Stable string for use as a Prometheus label value.
    pub fn as_label(self) -> &'static str {
        match self {
            KeySource::RoutingKey => "routing_key",
            KeySource::PromptCacheKey => "prompt_cache_key",
            KeySource::PromptPrefixCacheAware => "prompt_prefix_cache_aware",
            KeySource::PromptPrefix => "prompt_prefix",
            KeySource::SystemMessage => "system_message",
            KeySource::None => "none",
        }
    }
}

/// Resolved routing key. `hash` is `None` only when `source == None`.
#[derive(Debug, Clone)]
pub struct RoutingKeyResolved {
    pub hash: Option<u64>,
    pub source: KeySource,
    /// Set only when raw logging is opted in (`SIE_ROUTING_LOG_RAW=1`).
    /// Always `None` in production by construction. Read by the
    /// `SIE_ROUTING_LOG_RAW` startup-warning code path and by ops
    /// tooling; not read on the hot path so dead-code lint allowed.
    ///
    /// Only compiled in when the `raw-routing-logs` Cargo feature is
    /// enabled. In default release builds this field does not exist on
    /// the struct, so raw key strings cannot leak into any consumer
    /// (logs, metrics labels, debug dumps) regardless of env var.
    #[cfg(feature = "raw-routing-logs")]
    #[allow(dead_code)]
    pub raw_for_debug: Option<String>,
}

/// Stub trait the chat-completions surface implements.
/// The routing entry point only ships the trait so the priority cascade in
/// [`resolve_from_generate`] has a well-defined extension point.
#[allow(dead_code)]
pub trait MaybeChatRequest {
    /// Returns the system-message text used as a fallback routing key.
    fn extract_system_message(&self) -> Option<&str>;
}

/// Resolve a routing key for a `/v1/generate/...` (or `/v1/chat/...`) request.
///
/// This is the entry point used by [`crate::routing::pick_worker`].
/// Handles both prompt-shaped and chat-shaped inputs:
/// chat requests use the first ``system`` message (or the last user
/// message if no system message exists) as the prompt-prefix fallback.
pub fn resolve_from_generate(params: &GenerateParams) -> RoutingKeyResolved {
    let (prompt, system) = match &params.input {
        GenerateInput::Prompt { prompt } => (Some(prompt.as_str()), None),
        GenerateInput::Messages { messages } => {
            // Prefer the first ``system`` message as the routing-key
            // fallback; chat templates almost always have one and it's
            // typically stable across requests in the same conversation
            // (tenant-id, model-policy, etc.). Fall back to the last
            // user message so non-system chats still get a deterministic
            // pick.
            let system = messages
                .iter()
                .find(|m| m.role == "system")
                .map(|m| m.content.as_str());
            let last_user = messages
                .iter()
                .rev()
                .find(|m| m.role == "user")
                .map(|m| m.content.as_str());
            (last_user, system)
        }
    };
    resolve(
        params.routing_key.as_deref(),
        params.prompt_cache_key.as_deref(),
        prompt,
        system,
    )
}

/// Pure resolution helper — independent of `GenerateParams` so chat
/// and any future request shapes can reuse it. Reads the process-wide
/// [`PrefixConfig`]; [`resolve_with_prefix`] takes an explicit config for
/// tests.
pub fn resolve(
    routing_key: Option<&str>,
    prompt_cache_key: Option<&str>,
    prompt: Option<&str>,
    system_message: Option<&str>,
) -> RoutingKeyResolved {
    resolve_with_prefix(
        routing_key,
        prompt_cache_key,
        prompt,
        system_message,
        &PREFIX_CONFIG,
    )
}

/// As [`resolve`], with the cache-aware prefix config injected. Priority:
/// `routing_key` → `prompt_cache_key` → **cache-aware prefix** → legacy
/// system/prompt window → `None`. The caller-supplied `routing_key` /
/// `prompt_cache_key` keep their exact whole-string hash (precise override);
/// only the prompt/system fallback gains normalized prefix affinity.
pub fn resolve_with_prefix(
    routing_key: Option<&str>,
    prompt_cache_key: Option<&str>,
    prompt: Option<&str>,
    system_message: Option<&str>,
    cfg: &PrefixConfig,
) -> RoutingKeyResolved {
    // 1–2: caller-supplied exact keys (full-string hash).
    if let Some(s) = routing_key.filter(|s| !s.is_empty()) {
        return build_resolved(KeySource::RoutingKey, Some(hash_full_bytes(s)), Some(s));
    }
    if let Some(s) = prompt_cache_key.filter(|s| !s.is_empty()) {
        return build_resolved(KeySource::PromptCacheKey, Some(hash_full_bytes(s)), Some(s));
    }

    // 3: cache-aware prefix from the system message (preferred) else prompt.
    let prefix_source = system_message
        .filter(|s| !s.is_empty())
        .or_else(|| prompt.filter(|s| !s.is_empty()));
    if cfg.enabled {
        if let Some(src) = prefix_source {
            if let Some(h) = derive_prefix_key_with(
                src,
                cfg.prefix_bytes,
                cfg.min_bytes,
                cfg.byte_preserving,
                cfg.salt.as_deref(),
            ) {
                return build_resolved(KeySource::PromptPrefixCacheAware, Some(h), Some(src));
            }
        }
    }

    // 4: legacy whole-window fallback (short/unique prefix, or mode=off) —
    // identical to the prior behaviour, so never worse than today.
    if let Some(s) = system_message.filter(|s| !s.is_empty()) {
        return build_resolved(KeySource::SystemMessage, Some(hash_bytes(s)), Some(s));
    }
    if let Some(s) = prompt.filter(|s| !s.is_empty()) {
        return build_resolved(KeySource::PromptPrefix, Some(hash_bytes(s)), Some(s));
    }

    // 5: nothing to key on → caller falls through to round-robin / pool.
    build_resolved(KeySource::None, None, None)
}

fn build_resolved(source: KeySource, hash: Option<u64>, raw: Option<&str>) -> RoutingKeyResolved {
    #[cfg(feature = "raw-routing-logs")]
    let raw_for_debug = if super::log_raw_keys_enabled() {
        raw.map(|s| s.to_string())
    } else {
        None
    };
    #[cfg(not(feature = "raw-routing-logs"))]
    let _ = raw;
    RoutingKeyResolved {
        hash,
        source,
        #[cfg(feature = "raw-routing-logs")]
        raw_for_debug,
    }
}

/// Derive a cache-aware prefix key from a leading window of `source`.
///
/// Normalizes so cosmetic reformatting of an identical prefix still
/// co-locates: trim leading whitespace, lowercase, and collapse internal
/// whitespace runs to a single space. Returns `None` (caller falls back to
/// the legacy key) when the normalized prefix is shorter than `min_bytes` —
/// too short to be worth pinning. Otherwise hashes the first `prefix_bytes`
/// of the normalized form. The hash is opaque (never reconstructed), so a
/// multibyte char split at the window boundary is harmless: identical input
/// always yields an identical key.
///
/// This is the back-compat entry point; it calls
/// [`derive_prefix_key_with`] with `byte_preserving = false` and `salt = None`.
/// Kept primarily so existing tests and external callers retain the
/// unsalted-normalized signature; production code paths go through
/// [`derive_prefix_key_with`] via [`resolve_with_prefix`].
#[allow(dead_code)]
pub fn derive_prefix_key(source: &str, prefix_bytes: usize, min_bytes: usize) -> Option<u64> {
    derive_prefix_key_with(source, prefix_bytes, min_bytes, false, None)
}

/// As [`derive_prefix_key`], but with the byte-preserving and salt knobs
/// exposed. See the module-level docs for the semantics of each.
///
/// - `byte_preserving = false` (default): normalize before hashing
///   (lowercase + collapse internal whitespace runs). Identical prompts
///   with cosmetic-only differences co-locate.
/// - `byte_preserving = true`: hash the raw input bytes directly. The
///   routing key matches byte-for-byte what SGLang's radix cache sees;
///   whitespace-only-different prompts route to different workers.
/// - `salt`: optional tenant/namespace bytes mixed in before the prefix
///   bytes. Distinct salts produce distinct keys for the same prompt.
pub fn derive_prefix_key_with(
    source: &str,
    prefix_bytes: usize,
    min_bytes: usize,
    byte_preserving: bool,
    salt: Option<&str>,
) -> Option<u64> {
    // Pick the bytes to hash. In byte-preserving mode we feed the raw
    // input straight through; in the default mode we normalize first so
    // cosmetic reformatting of the same prefix collapses to one key.
    let normalized_storage: String;
    let bytes: &[u8] = if byte_preserving {
        let raw = source.as_bytes();
        if raw.len() < min_bytes {
            return None;
        }
        raw
    } else {
        let mut normalized =
            String::with_capacity(source.len().min(prefix_bytes.saturating_mul(2)));
        let mut last_was_space = false;
        for ch in source.trim_start().chars() {
            if ch.is_whitespace() {
                if !last_was_space {
                    normalized.push(' ');
                    last_was_space = true;
                }
            } else {
                normalized.extend(ch.to_lowercase());
                last_was_space = false;
            }
        }
        // trim_end gives us a &str borrowed from `normalized`; copy the
        // trimmed slice into our owned storage so the borrow lifetime
        // matches the outer `bytes` binding.
        let trimmed_len = normalized.trim_end().len();
        normalized.truncate(trimmed_len);
        if normalized.len() < min_bytes {
            return None;
        }
        normalized_storage = normalized;
        normalized_storage.as_bytes()
    };
    let n = bytes.len().min(prefix_bytes);
    // Mix the salt in before the prefix bytes. We hash the concatenation
    // `salt || 0x1F || prefix` so a salt change perturbs the entire
    // resulting key (xxh3 is not commutative). The 0x1F separator (US,
    // unit separator) prevents accidental collisions between e.g.
    // (salt="a", prefix="bc") and (salt="ab", prefix="c"). When `salt`
    // is `None` we hash exactly the prefix bytes, preserving the legacy
    // (unsalted) key value bit-for-bit.
    match salt {
        Some(s) if !s.is_empty() => {
            let mut buf = Vec::with_capacity(s.len() + 1 + n);
            buf.extend_from_slice(s.as_bytes());
            buf.push(0x1F);
            buf.extend_from_slice(&bytes[..n]);
            Some(xxh3_64(&buf))
        }
        _ => Some(xxh3_64(&bytes[..n])),
    }
}

/// xxh3 64-bit, truncated to the first [`PROMPT_PREFIX_BYTES`] of the
/// UTF-8 encoded input. Truncation is byte-wise; this is fine because
/// the hash is opaque (we never reconstruct the string from it).
pub fn hash_bytes(s: &str) -> u64 {
    let bytes = s.as_bytes();
    let n = bytes.len().min(PROMPT_PREFIX_BYTES);
    xxh3_64(&bytes[..n])
}

fn hash_full_bytes(s: &str) -> u64 {
    xxh3_64(s.as_bytes())
}
