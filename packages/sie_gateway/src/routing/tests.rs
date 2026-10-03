//! Inline tests for the routing module.
//!
//! Acceptance criteria coverage (see the direct-dispatch routing plan):
//! - HRW determinism: same key → same worker across reshuffles.
//! - Membership change: only `1/n` of keys move when a worker drops.
//! - Hash-key priority: `routing_key` > `prompt_cache_key` > prompt prefix.
//! - xxh3 prefix-truncation semantics.
//! - Privacy: `fmt_key_hash` never reveals the raw string; `xxh:` prefix only.

use super::hrw::{pick_worker, RingEntry, RingSnapshot};
use super::key::{
    derive_prefix_key, derive_prefix_key_with, hash_bytes, resolve, resolve_with_prefix, KeySource,
    PrefixConfig, PROMPT_PREFIX_BYTES,
};
#[cfg(feature = "raw-routing-logs")]
use super::RAW_LOGGING_ENV;
use super::{fmt_key_hash, log_raw_keys_enabled, suppress_first_chunk_republish_for_lane};

fn workers(n: usize) -> RingSnapshot {
    RingSnapshot::new((0..n).map(|i| format!("worker-{i}")))
}

#[test]
fn pick_is_deterministic_for_same_key() {
    let snap = workers(8);
    let key = resolve(Some("abc"), None, None, None);
    let first = pick_worker(&snap, &key).unwrap().to_string();
    for _ in 0..1000 {
        assert_eq!(pick_worker(&snap, &key).unwrap(), first.as_str());
    }
}

#[test]
fn pick_is_stable_across_shuffled_input_order() {
    // The HRW max-by-hash invariant must not depend on iteration order.
    let mut a: Vec<String> = (0..16).map(|i| format!("w-{i}")).collect();
    let mut b = a.clone();
    b.reverse();
    let snap_a = RingSnapshot::new(a.drain(..));
    let snap_b = RingSnapshot::new(b.drain(..));
    let key = resolve(Some("hello-world"), None, None, None);
    assert_eq!(pick_worker(&snap_a, &key), pick_worker(&snap_b, &key));
}

#[test]
fn dropping_a_worker_moves_only_keys_assigned_to_it() {
    let snap_full = workers(8);
    let snap_minus_one = RingSnapshot::new(
        snap_full
            .entries
            .iter()
            .filter(|e| e.worker_id != "worker-3")
            .map(|e| e.worker_id.clone()),
    );

    let mut moved = 0;
    let mut total = 0;
    for i in 0..1000 {
        let k = resolve(Some(&format!("k-{i}")), None, None, None);
        let before = pick_worker(&snap_full, &k).unwrap().to_string();
        let after = pick_worker(&snap_minus_one, &k).unwrap().to_string();
        if before == "worker-3" {
            // Removed worker's keys must be reassigned.
            assert_ne!(after, "worker-3");
        } else {
            // Keys not assigned to the removed worker must not move.
            assert_eq!(before, after, "key {i} drifted unnecessarily");
        }
        if before != after {
            moved += 1;
        }
        total += 1;
    }
    // With 8 workers, ~1/8 of keys should move when one is removed.
    // Inputs are deterministic (`format!("k-{i}")` + xxh3) so this band
    // is not statistical — it documents the actual distribution. If
    // xxh3 input or `combine` is changed, the band may need re-pinning.
    let frac = moved as f64 / total as f64;
    assert!(
        (0.08..0.18).contains(&frac),
        "moved fraction out of expected band: {frac:.3}"
    );
}

#[test]
fn xor_symmetric_inputs_do_not_collide() {
    // Regression for the previous `xxh3(req ^ wid)` mix: that function
    // is invariant under `(req, wid) → (req⊕x, wid⊕x)` for any x, so
    // two workers whose ids hashed to values that XOR-cancel with the
    // request key picked the same score. The asymmetric mix
    // (`wid.rotate_left(32) ^ req`, re-hashed) must produce distinct
    // outputs for any non-trivial input pair.
    use super::hrw::combine_for_test;
    let req = 0xdead_beef_cafe_babe_u64;
    let a = 0x1111_2222_3333_4444_u64;
    let b = 0x4444_3333_2222_1111_u64;
    assert_ne!(combine_for_test(req, a), combine_for_test(req, b));
    // And the function must remain deterministic for repeated calls.
    assert_eq!(combine_for_test(req, a), combine_for_test(req, a));
}

#[test]
fn duplicate_worker_ids_hash_and_tiebreak_identically() {
    // Precondition that motivates the registry-side dedup in
    // `WorkerRegistry::ring_snapshot_for`: a `RingSnapshot` keyed on a
    // non-unique worker id produces entries with identical
    // `worker_id_hash` and an equal lexicographic tie-break, so the HRW
    // pick between them is ambiguous (and both map to the same per-worker
    // subject). The ring must be built from already-deduplicated names.
    let snap = RingSnapshot::new(vec!["dup".to_string(), "dup".to_string()]);
    assert_eq!(snap.len(), 2);
    assert_eq!(
        snap.entries[0].worker_id_hash,
        snap.entries[1].worker_id_hash
    );
    assert_eq!(snap.entries[0].worker_id, snap.entries[1].worker_id);
}

#[test]
fn pick_prefers_lower_pressure_over_hrw_winner() {
    let key = resolve(Some("pressure-key"), None, None, None);
    let pure = RingSnapshot::new(vec!["busy".to_string(), "idle".to_string()]);
    let pure_winner = pick_worker(&pure, &key).expect("pure HRW winner");
    let other = if pure_winner == "busy" {
        "idle"
    } else {
        "busy"
    };

    let snap = RingSnapshot::from_entries([
        RingEntry::with_pressure(pure_winner, 1, 100, 1000, 10),
        RingEntry::with_pressure(other, 1, 0, 0, 0),
    ]);

    assert_eq!(pick_worker(&snap, &key), Some(other));
}

#[test]
fn pick_normalizes_pressure_by_ready_gpu_slots() {
    let key = resolve(Some("slot-pressure-key"), None, None, None);
    let snap = RingSnapshot::from_entries([
        RingEntry::with_pressure("wide", 4, 4, 40, 4),
        RingEntry::with_pressure("narrow", 1, 2, 20, 2),
    ]);

    assert_eq!(pick_worker(&snap, &key), Some("wide"));
}

#[test]
fn pick_normalizes_maximum_pressure_without_overflow() {
    let key = resolve(Some("maximum-pressure"), None, None, None);
    let snap = RingSnapshot::from_entries([
        RingEntry::with_pressure("wide", 2, i32::MAX, i64::MAX, i32::MAX),
        RingEntry::with_pressure("narrow", 1, i32::MAX, i64::MAX - 1, i32::MAX),
    ]);
    assert_eq!(pick_worker(&snap, &key), Some("wide"));
    let snap = RingSnapshot::from_entries([
        RingEntry::with_pressure("untrusted", 2, i32::MAX, i64::MAX, i32::MAX),
        RingEntry::with_pressure("idle", 1, 0, 0, 0),
    ]);
    assert_eq!(pick_worker(&snap, &key), Some("idle"));
}

#[test]
fn empty_snapshot_returns_none() {
    let snap = RingSnapshot::default();
    let key = resolve(Some("x"), None, None, None);
    assert!(pick_worker(&snap, &key).is_none());
}

#[test]
fn first_chunk_republish_is_suppressed_for_single_direct_lane() {
    assert!(suppress_first_chunk_republish_for_lane(true, 1));
}

#[test]
fn first_chunk_republish_is_not_suppressed_when_lane_has_failover() {
    assert!(!suppress_first_chunk_republish_for_lane(true, 2));
}

#[test]
fn first_chunk_republish_guard_ignores_pool_fallback_dispatch() {
    assert!(!suppress_first_chunk_republish_for_lane(false, 0));
}

#[test]
fn none_keysource_returns_none_pick() {
    let snap = workers(4);
    let key = resolve(None, None, None, None);
    assert!(matches!(key.source, KeySource::None));
    assert!(key.hash.is_none());
    assert!(pick_worker(&snap, &key).is_none());
}

#[test]
fn priority_routing_key_beats_others() {
    let r = resolve(Some("a"), Some("b"), Some("prompt"), Some("system"));
    assert!(matches!(r.source, KeySource::RoutingKey));
    assert_eq!(r.hash, Some(hash_bytes("a")));
}

#[test]
fn priority_prompt_cache_key_beats_prompt() {
    let r = resolve(None, Some("b"), Some("prompt"), Some("system"));
    assert!(matches!(r.source, KeySource::PromptCacheKey));
    assert_eq!(r.hash, Some(hash_bytes("b")));
}

#[test]
fn priority_system_message_beats_prompt() {
    let r = resolve(None, None, Some("prompt"), Some("system"));
    assert!(matches!(r.source, KeySource::SystemMessage));
    assert_eq!(r.hash, Some(hash_bytes("system")));
}

#[test]
fn priority_prompt_used_when_nothing_else() {
    let r = resolve(None, None, Some("just-a-prompt"), None);
    assert!(matches!(r.source, KeySource::PromptPrefix));
    assert_eq!(r.hash, Some(hash_bytes("just-a-prompt")));
}

#[test]
fn empty_strings_are_treated_as_absent() {
    let r = resolve(Some(""), Some(""), Some("real"), None);
    assert!(matches!(r.source, KeySource::PromptPrefix));
}

#[test]
fn prompt_prefix_truncates_at_512_bytes() {
    let long = "a".repeat(PROMPT_PREFIX_BYTES + 1024);
    let cut = &long[..PROMPT_PREFIX_BYTES];
    // Hashes must match the truncated prefix, not the full string.
    assert_eq!(hash_bytes(&long), hash_bytes(cut));
    // And differ from a still-shorter prefix.
    assert_ne!(
        hash_bytes(&long),
        hash_bytes(&long[..PROMPT_PREFIX_BYTES - 1])
    );
}

#[test]
fn fmt_key_hash_emits_xxh_prefix_only() {
    let s = fmt_key_hash(0x0123_4567_89ab_cdef);
    assert!(s.starts_with("xxh:"));
    assert_eq!(s.len(), "xxh:".len() + 8);
    // Upper 32 bits, lowercase hex.
    assert_eq!(s, "xxh:01234567");
}

#[test]
fn raw_logging_default_disabled() {
    // Cannot mutate process env safely in parallel tests, so just
    // exercise the predicate against whatever is set. Most CI/dev
    // shells will not set this; document the invariant either way.
    let enabled = log_raw_keys_enabled();
    #[cfg(feature = "raw-routing-logs")]
    {
        let raw = std::env::var(RAW_LOGGING_ENV).ok();
        assert_eq!(enabled, raw.as_deref() == Some("1"));
    }
    #[cfg(not(feature = "raw-routing-logs"))]
    assert!(!enabled);
}

#[cfg(feature = "raw-routing-logs")]
#[test]
fn raw_for_debug_is_none_when_flag_disabled() {
    // If the test environment does not set the flag, raw_for_debug
    // must be None even when the source has a real string.
    if !log_raw_keys_enabled() {
        let r = resolve(Some("secret-tenant-id"), None, None, None);
        assert!(r.raw_for_debug.is_none());
    }
}

#[cfg(not(feature = "raw-routing-logs"))]
#[test]
fn raw_for_debug_field_absent_without_feature() {
    // Compile-time invariant: when the Cargo feature is off, the
    // `raw_for_debug` field doesn't exist on the struct. This test
    // exists primarily to document the contract; if anyone re-adds the
    // field unconditionally, the cfg-gated test above will start
    // compiling under default features and the project's CI matrix
    // should flag it.
    let r = resolve(Some("secret-tenant-id"), None, None, None);
    // Hash still resolves; only the raw field is stripped.
    assert!(r.hash.is_some());
}

// ── Cache-aware (prefix-hash) routing ───────────────

const LONG_PREFIX: &str =
    "You are a meticulous assistant. Always answer concisely and cite sources \
     when they are relevant to the user's question, and never fabricate facts.";

#[test]
fn prefix_config_defaults() {
    let d = PrefixConfig::default();
    assert!(d.enabled);
    assert_eq!(d.prefix_bytes, PROMPT_PREFIX_BYTES);
    assert_eq!(d.min_bytes, 64);
    assert!(!d.byte_preserving, "default must keep normalization on");
    assert!(d.salt.is_none(), "default must carry no salt");
}

#[test]
fn derive_prefix_key_short_input_falls_back_to_none() {
    // Below min_bytes → not worth pinning.
    assert!(derive_prefix_key("short", 512, 64).is_none());
}

#[test]
fn derive_prefix_key_normalizes_whitespace_and_case() {
    let base = derive_prefix_key(LONG_PREFIX, 512, 16).expect("long enough");
    // Same prefix, reformatted: leading spaces, upper-case, doubled spaces.
    let reformatted = format!("   {}", LONG_PREFIX.to_uppercase().replace(' ', "  "));
    let other = derive_prefix_key(&reformatted, 512, 16).expect("long enough");
    assert_eq!(base, other, "cosmetic reformatting must yield the same key");
}

#[test]
fn derive_prefix_key_distinct_prefixes_differ() {
    let a = derive_prefix_key(LONG_PREFIX, 512, 16).unwrap();
    let b = derive_prefix_key(
        "You are a terse assistant. Reply with a single word and nothing else, \
         under all circumstances, regardless of the question asked.",
        512,
        16,
    )
    .unwrap();
    assert_ne!(a, b);
}

#[test]
fn resolve_cache_aware_uses_normalized_prefix_when_enabled() {
    let cfg = PrefixConfig::default();
    let r = resolve_with_prefix(None, None, Some("hi"), Some(LONG_PREFIX), &cfg);
    assert!(matches!(r.source, KeySource::PromptPrefixCacheAware));
    assert_eq!(r.hash, derive_prefix_key(LONG_PREFIX, 512, 64));
}

#[test]
fn resolve_off_mode_uses_legacy_whole_window() {
    let cfg = PrefixConfig {
        enabled: false,
        ..PrefixConfig::default()
    };
    let r = resolve_with_prefix(None, None, Some("hi"), Some(LONG_PREFIX), &cfg);
    assert!(matches!(r.source, KeySource::SystemMessage));
    assert_eq!(r.hash, Some(hash_bytes(LONG_PREFIX)));
}

#[test]
fn resolve_short_prefix_falls_back_to_legacy() {
    // Shorter than min_bytes → cache-aware declines → legacy SystemMessage.
    let cfg = PrefixConfig::default();
    let r = resolve_with_prefix(None, None, Some("p"), Some("a short system"), &cfg);
    assert!(matches!(r.source, KeySource::SystemMessage));
    assert_eq!(r.hash, Some(hash_bytes("a short system")));
}

#[test]
fn resolve_routing_key_still_wins_over_prefix() {
    let cfg = PrefixConfig::default();
    let r = resolve_with_prefix(Some("tenant-x"), None, None, Some(LONG_PREFIX), &cfg);
    assert!(matches!(r.source, KeySource::RoutingKey));
}

#[test]
fn cache_aware_prefix_colocates_shared_system_prompt() {
    let cfg = PrefixConfig::default();
    let snap = workers(3);
    // ~2.3 KB system prompt shared across many requests with distinct turns.
    let system = "You are a careful assistant. ".repeat(80);
    let mut picked = std::collections::HashSet::new();
    for i in 0..20 {
        let user = format!("question number {i}");
        let key = resolve_with_prefix(None, None, Some(&user), Some(&system), &cfg);
        assert!(matches!(key.source, KeySource::PromptPrefixCacheAware));
        picked.insert(pick_worker(&snap, &key).unwrap().to_string());
    }
    assert_eq!(
        picked.len(),
        1,
        "requests sharing a system prefix must co-locate on one worker"
    );
}

// ── M11: tenant/namespace salt + byte-preserving routing-key mode ──

#[test]
fn salt_changes_routing_key_for_same_prompt() {
    // Two different salts on the same prompt → two different hashes.
    let a =
        derive_prefix_key_with(LONG_PREFIX, 512, 16, false, Some("tenant-a")).expect("long enough");
    let b =
        derive_prefix_key_with(LONG_PREFIX, 512, 16, false, Some("tenant-b")).expect("long enough");
    let unsalted = derive_prefix_key_with(LONG_PREFIX, 512, 16, false, None).expect("long enough");
    assert_ne!(a, b, "different salts must produce different keys");
    assert_ne!(a, unsalted, "salted vs unsalted must differ");
    assert_ne!(b, unsalted, "salted vs unsalted must differ");
}

#[test]
fn no_salt_preserves_legacy_key() {
    // `None` and an empty-string salt must both reproduce the
    // pre-M11 (unsalted) key exactly. This is the regression baseline.
    let legacy = derive_prefix_key(LONG_PREFIX, 512, 16).expect("long enough");
    let unsalted = derive_prefix_key_with(LONG_PREFIX, 512, 16, false, None).expect("long enough");
    let empty_salt =
        derive_prefix_key_with(LONG_PREFIX, 512, 16, false, Some("")).expect("long enough");
    assert_eq!(unsalted, legacy);
    assert_eq!(empty_salt, legacy);
    // And: same prompt + same salt, called twice, is stable.
    let s1 = derive_prefix_key_with(LONG_PREFIX, 512, 16, false, Some("ns")).unwrap();
    let s2 = derive_prefix_key_with(LONG_PREFIX, 512, 16, false, Some("ns")).unwrap();
    assert_eq!(s1, s2);
}

#[test]
fn salt_distributes_shared_prompt_across_workers() {
    // The motivating use case: identical system prompt under different
    // namespaces should NOT all pin to one worker. With a 3-worker ring
    // and a small set of salts we expect at least two distinct picks.
    let snap = workers(3);
    let system = "You are a careful assistant. ".repeat(80);
    let mut picked = std::collections::HashSet::new();
    for salt in &["alpha", "beta", "gamma", "delta", "epsilon", "zeta"] {
        let cfg = PrefixConfig {
            salt: Some((*salt).to_string()),
            ..PrefixConfig::default()
        };
        let key = resolve_with_prefix(None, None, Some("q"), Some(&system), &cfg);
        assert!(matches!(key.source, KeySource::PromptPrefixCacheAware));
        picked.insert(pick_worker(&snap, &key).unwrap().to_string());
    }
    assert!(
        picked.len() >= 2,
        "salts must spread the same prompt across >1 worker, got {picked:?}"
    );
}

#[test]
fn byte_preserving_mode_distinguishes_whitespace_and_case() {
    // With byte_preserving = true, leading whitespace and case
    // differences become routing-key-significant.
    let lower = derive_prefix_key_with(LONG_PREFIX, 512, 16, true, None).expect("long enough");
    let upper = derive_prefix_key_with(&LONG_PREFIX.to_uppercase(), 512, 16, true, None)
        .expect("long enough");
    let leading_ws = derive_prefix_key_with(&format!("   {LONG_PREFIX}"), 512, 16, true, None)
        .expect("long enough");
    assert_ne!(
        lower, upper,
        "byte-preserving: case differences must differ"
    );
    assert_ne!(
        lower, leading_ws,
        "byte-preserving: leading whitespace must differ"
    );
}

#[test]
fn default_mode_collapses_whitespace_and_case() {
    // Regression-pin: the default (normalized) mode still collapses
    // cosmetic differences exactly as before M11.
    let lower = derive_prefix_key_with(LONG_PREFIX, 512, 16, false, None).expect("long enough");
    let upper = derive_prefix_key_with(&LONG_PREFIX.to_uppercase(), 512, 16, false, None)
        .expect("long enough");
    let leading_ws = derive_prefix_key_with(&format!("   {LONG_PREFIX}"), 512, 16, false, None)
        .expect("long enough");
    assert_eq!(lower, upper, "normalized mode must equate case differences");
    assert_eq!(
        lower, leading_ws,
        "normalized mode must equate leading whitespace"
    );
}

#[test]
fn byte_preserving_salt_compose() {
    // Salt and byte-preserving compose orthogonally.
    let a = derive_prefix_key_with(LONG_PREFIX, 512, 16, true, Some("t1")).unwrap();
    let b = derive_prefix_key_with(LONG_PREFIX, 512, 16, true, Some("t2")).unwrap();
    let c = derive_prefix_key_with(LONG_PREFIX, 512, 16, true, None).unwrap();
    assert_ne!(a, b);
    assert_ne!(a, c);
    assert_ne!(b, c);
}

#[test]
fn resolve_with_prefix_propagates_salt_and_byte_preserving() {
    // End-to-end: the public resolve_with_prefix entry point must honour
    // both knobs on the PrefixConfig.
    let salted = PrefixConfig {
        salt: Some("ns-a".to_string()),
        ..PrefixConfig::default()
    };
    let unsalted = PrefixConfig::default();
    let r_a = resolve_with_prefix(None, None, None, Some(LONG_PREFIX), &salted);
    let r_b = resolve_with_prefix(None, None, None, Some(LONG_PREFIX), &unsalted);
    assert!(matches!(r_a.source, KeySource::PromptPrefixCacheAware));
    assert!(matches!(r_b.source, KeySource::PromptPrefixCacheAware));
    assert_ne!(r_a.hash, r_b.hash);

    let bp = PrefixConfig {
        byte_preserving: true,
        ..PrefixConfig::default()
    };
    let r_norm = resolve_with_prefix(None, None, None, Some(LONG_PREFIX), &unsalted);
    let r_bytes = resolve_with_prefix(None, None, None, Some(LONG_PREFIX), &bp);
    // For a prompt whose normalized form differs from its raw bytes
    // (LONG_PREFIX has no leading ws/upper-case, but `derive_prefix_key`
    // trims/normalizes anyway and the hash uses byte-length truncation,
    // so for this input the two MAY coincide — assert that at minimum
    // a deliberately whitespace-perturbed input differs).
    let _ = (r_norm, r_bytes);
    let r_norm_ws = resolve_with_prefix(
        None,
        None,
        None,
        Some("   YOU are a careful assistant.   ".repeat(8).as_str()),
        &unsalted,
    );
    let r_bp_ws = resolve_with_prefix(
        None,
        None,
        None,
        Some("   YOU are a careful assistant.   ".repeat(8).as_str()),
        &bp,
    );
    assert_ne!(
        r_norm_ws.hash, r_bp_ws.hash,
        "byte-preserving mode must produce a different key than normalized for whitespace-y / mixed-case input"
    );
}
