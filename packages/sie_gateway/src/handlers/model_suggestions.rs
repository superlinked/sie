//! Hints for the `MODEL_NOT_FOUND` response: "was retired" or "Did you mean".
//!
//! The hint is a function of exactly three inputs: the string the caller sent,
//! the operator's retired-model map, and the model ids that caller may list.
//! Nothing resolved from the request (an alias target, a canonical id, a
//! governed route) takes part, and the lookup never asks whether the requested
//! id itself exists, so a model hidden by a
//! [`ModelAccessPolicy`](crate::server::ModelAccessPolicy), a model that does
//! not exist, and a model whose route was never compiled still produce the same
//! body for the same request string.
//!
//! A retired id has no registry entry, so its canonical case is unknown. The
//! retired note therefore needs the caller to be allowed to list the retired
//! id both in the operator's spelling (with `__` read as `/`) and in lower
//! case, plus the replacement as the registry spells it.

use std::collections::HashMap;

use crate::config::RetiredModel;
use crate::server::AppState;

/// Most ids one response suggests.
const MAX_SUGGESTIONS: usize = 3;

/// Shortest normalized request that may match by prefix or by edit distance.
/// Below this, almost every id is "close".
const MIN_FUZZY_LEN: usize = 4;

/// Requests longer than this get no hint, which bounds the work one 404 can
/// cost.
const MAX_REQUEST_LEN: usize = 256;

/// Largest edit distance accepted as a typo, whatever the id length.
const MAX_TYPO_DISTANCE: usize = 3;

/// One accepted edit per this many normalized characters.
const TYPO_CHARS_PER_EDIT: usize = 6;

/// What the `MODEL_NOT_FOUND` message adds after "not found".
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum NotFoundHint {
    /// "Did you mean" ids, closest first. Empty means no hint.
    Suggestions(Vec<String>),
    /// The operator retired the requested id in favour of `replacement`.
    Retired { replacement: String },
}

/// The `MODEL_NOT_FOUND` message for `requested` with its `hint`.
///
/// `requested` is always the caller's own string.
pub(crate) fn unknown_model_message(requested: &str, hint: &NotFoundHint) -> String {
    let suggestions = match hint {
        NotFoundHint::Retired { replacement } => {
            return format!("Model '{requested}' was retired. Use '{replacement}'.");
        }
        NotFoundHint::Suggestions(suggestions) => suggestions.as_slice(),
    };
    match suggestions {
        [] => format!("Model '{requested}' not found"),
        [only] => format!("Model '{requested}' not found. Did you mean '{only}'?"),
        many => {
            let quoted: Vec<String> = many.iter().map(|id| format!("'{id}'")).collect();
            format!(
                "Model '{requested}' not found. Did you mean one of: {}?",
                quoted.join(", ")
            )
        }
    }
}

/// The hint every `MODEL_NOT_FOUND` site renders for `requested`.
///
/// A retired note wins when the operator retired the caller's string and this
/// caller may list the retired id in the operator's spelling and in lower
/// case, plus the replacement as the registry spells it. Otherwise the
/// "Did you mean" ids, drawn only from what this caller may list: the
/// registry's ids that pass the visibility filter `GET /v1/models` applies.
/// The listing can show more: ids only a worker advertises are left out,
/// because a request for one gets this same 404 from a populated registry.
///
/// An alias with the same name as a retired id wins on its own, because the
/// request then resolves and never reaches this 404.
pub(crate) fn not_found_hint(
    state: &AppState,
    ext: &axum::http::Extensions,
    requested: &str,
) -> NotFoundHint {
    let visible = |id: &str| {
        state
            .model_access_policy
            .as_ref()
            .is_none_or(|policy| policy.visible(id, ext))
    };
    hint_for(
        requested,
        &state.config.retired_models,
        state.model_registry.list_models(),
        visible,
    )
}

/// [`not_found_hint`] over plain inputs: `listed` is every registry id and
/// `visible` says which of them this caller may list.
///
/// The retired id is asked about twice, as the operator spelled it (`__` read
/// as `/`) and lower-cased (the map key), because it has no registry entry
/// to fix its canonical case: a case-sensitive policy must allow both. The
/// second call is skipped when the two forms are equal.
fn hint_for(
    requested: &str,
    retired: &HashMap<String, RetiredModel>,
    listed: Vec<String>,
    visible: impl Fn(&str) -> bool,
) -> NotFoundHint {
    if let Some((key, entry)) = retired_entry(retired, requested) {
        // Never the `__` spelling: the policy contract takes `/`.
        if let Some(replacement) = registered_replacement(&listed, &entry.replacement) {
            if visible(&entry.declared)
                && (key == entry.declared || visible(key))
                && visible(replacement)
            {
                return NotFoundHint::Retired {
                    replacement: replacement.to_string(),
                };
            }
        }
    }
    // Variants are never suggested, so the policy is not asked about them.
    let candidates = listed
        .into_iter()
        .filter(|id| !id.contains(':'))
        .filter(|id| visible(id));
    NotFoundHint::Suggestions(suggest_model_ids(requested, candidates))
}

/// The registry's spelling of `replacement`: the id that matches the whole
/// string ignoring case and reading `__` as `/`, as the gateway resolves ids.
/// A `model:profile` replacement matches only that registered variant, and a
/// base id never matches a variant. If several ids match, the byte-order
/// first wins.
fn registered_replacement<'a>(listed: &'a [String], replacement: &str) -> Option<&'a str> {
    let wanted = replacement.replace("__", "/").to_ascii_lowercase();
    listed
        .iter()
        .filter(|id| id.replace("__", "/").to_ascii_lowercase() == wanted)
        .map(String::as_str)
        .min()
}

/// The retired-map entry for `requested`, matched exactly on
/// [`retired_model_key`]. Never fuzzy.
fn retired_entry<'a>(
    retired: &'a HashMap<String, RetiredModel>,
    requested: &str,
) -> Option<(&'a str, &'a RetiredModel)> {
    if retired.is_empty() || requested.len() > MAX_REQUEST_LEN {
        return None;
    }
    retired
        .get_key_value(&retired_model_key(requested))
        .map(|(key, entry)| (key.as_str(), entry))
}

/// A retired id in the operator's case and in canonical form: trimmed,
/// without a `bundle:/` prefix or `:profile` suffix, `__` read as `/`. This
/// is the form the access policy is asked about.
pub(crate) fn retired_model_declared(id: &str) -> String {
    model_part(id.trim()).replace("__", "/")
}

/// The key a retired-model entry is stored and looked up under: the model id
/// without a `bundle:/` prefix or `:profile` suffix, lower case, `__` read as
/// `/`. Other separators are kept, so only the same id matches.
pub(crate) fn retired_model_key(id: &str) -> String {
    retired_model_declared(id).to_ascii_lowercase()
}

/// Up to [`MAX_SUGGESTIONS`] ids from `candidates` that `requested` most
/// likely meant, closest first. Empty when nothing is close.
///
/// A registry also lists `model:profile` variants. Those are skipped: the
/// request is matched without its own `:profile`, and naming every variant of
/// one model would crowd out other models.
pub(crate) fn suggest_model_ids(
    requested: &str,
    candidates: impl IntoIterator<Item = String>,
) -> Vec<String> {
    if requested.len() > MAX_REQUEST_LEN {
        return Vec::new();
    }
    let wanted = normalize(model_part(requested));
    if wanted.is_empty() {
        return Vec::new();
    }
    let mut scored: Vec<(Closeness, String)> = candidates
        .into_iter()
        .filter(|id| !id.contains(':'))
        .filter_map(|id| closeness(&wanted, &normalize(&id)).map(|score| (score, id)))
        .collect();
    scored.sort();
    scored.dedup_by(|a, b| a.1 == b.1);
    scored.truncate(MAX_SUGGESTIONS);
    scored.into_iter().map(|(_, id)| id).collect()
}

/// How a candidate matched, best first; the `usize` breaks ties inside a kind.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Closeness {
    /// Same id once case and separators are ignored.
    SameId,
    /// Same model name; the organization is missing or different.
    SameName,
    /// The request is the start of the id; fewer missing characters first.
    Prefix(usize),
    /// A few edits apart; fewer edits first.
    Typo(usize),
}

fn closeness(wanted: &str, candidate: &str) -> Option<Closeness> {
    if wanted == candidate {
        return Some(Closeness::SameId);
    }
    let candidate_name = name_part(candidate);
    let wanted_name = name_part(wanted);
    if wanted_name.len() >= MIN_FUZZY_LEN && wanted_name == candidate_name {
        return Some(Closeness::SameName);
    }
    // A request without an organization is compared with the model name alone.
    let target = if wanted.contains('/') {
        candidate
    } else {
        candidate_name
    };
    if wanted.len() < MIN_FUZZY_LEN {
        return None;
    }
    if target.starts_with(wanted) {
        return Some(Closeness::Prefix(target.len() - wanted.len()));
    }
    let allowed = (wanted.len() / TYPO_CHARS_PER_EDIT).clamp(1, MAX_TYPO_DISTANCE);
    if wanted.len().abs_diff(target.len()) > allowed {
        return None;
    }
    let distance = edit_distance(wanted.as_bytes(), target.as_bytes());
    (distance <= allowed).then_some(Closeness::Typo(distance))
}

/// The model id inside a request string: drops a leading `bundle:/` and a
/// trailing `:profile`. Purely syntactic, so it never consults the registry.
fn model_part(requested: &str) -> &str {
    let without_bundle = requested
        .split_once(":/")
        .map_or(requested, |(_, rest)| rest);
    without_bundle
        .split_once(':')
        .map_or(without_bundle, |(model, _)| model)
}

/// Lower case, `__` read as `/`, and every other separator dropped, so ids
/// that differ only in case or punctuation compare equal.
fn normalize(id: &str) -> String {
    id.replace("__", "/")
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '/')
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

fn name_part(normalized: &str) -> &str {
    normalized
        .rsplit_once('/')
        .map_or(normalized, |(_, name)| name)
}

/// Levenshtein distance over bytes; inputs are ASCII after [`normalize`].
fn edit_distance(a: &[u8], b: &[u8]) -> usize {
    let mut previous: Vec<usize> = (0..=b.len()).collect();
    let mut current = vec![0; b.len() + 1];
    for (i, &left) in a.iter().enumerate() {
        current[0] = i + 1;
        for (j, &right) in b.iter().enumerate() {
            let substitution = previous[j] + usize::from(left != right);
            current[j + 1] = substitution.min(previous[j + 1] + 1).min(current[j] + 1);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn catalog() -> Vec<String> {
        [
            "BAAI/bge-m3",
            "BAAI/bge-reranker-v2-m3",
            "Qwen/Qwen3.5-4B",
            "Qwen/Qwen3.6-27B",
            "Qwen/Qwen3.8-27B-FP8",
            "google/gemma-4-E4B-it",
        ]
        .into_iter()
        .map(String::from)
        .collect()
    }

    fn suggest(requested: &str) -> Vec<String> {
        suggest_model_ids(requested, catalog())
    }

    #[test]
    fn a_missing_organization_suggests_the_full_id() {
        assert_eq!(suggest("bge-m3"), ["BAAI/bge-m3"]);
    }

    #[test]
    fn a_wrong_organization_suggests_the_id_with_the_same_name() {
        assert_eq!(suggest("bge/bge-m3"), ["BAAI/bge-m3"]);
    }

    #[test]
    fn a_missing_suffix_suggests_the_longer_id_first() {
        // The id that only adds a suffix ranks ahead of a sibling one edit away.
        assert_eq!(
            suggest("Qwen/Qwen3.8-27B"),
            ["Qwen/Qwen3.8-27B-FP8", "Qwen/Qwen3.6-27B"]
        );
    }

    #[test]
    fn a_typo_suggests_the_nearest_id() {
        assert_eq!(suggest("BAAI/bge-n3"), ["BAAI/bge-m3"]);
        assert_eq!(suggest("gogle/gemma-4-E4B-it"), ["google/gemma-4-E4B-it"]);
    }

    #[test]
    fn case_and_separator_differences_suggest_the_canonical_id() {
        assert_eq!(suggest("baai/BGE_M3"), ["BAAI/bge-m3"]);
        assert_eq!(suggest("BAAI__bge.m3"), ["BAAI/bge-m3"]);
    }

    #[test]
    fn a_bundle_prefix_and_a_profile_suffix_are_ignored_when_matching() {
        assert_eq!(suggest("default:/bge-m3"), ["BAAI/bge-m3"]);
        assert_eq!(suggest("BAAI/bge-n3:banking"), ["BAAI/bge-m3"]);
    }

    #[test]
    fn profile_variants_of_a_model_are_not_suggested() {
        let listed = ["BAAI/bge-m3", "BAAI/bge-m3:dense", "BAAI/bge-m3:candle"]
            .into_iter()
            .map(String::from);
        assert_eq!(suggest_model_ids("bge-m3", listed), ["BAAI/bge-m3"]);
    }

    #[test]
    fn nothing_close_suggests_nothing() {
        assert!(suggest("acme/unrelated-model").is_empty());
        assert!(suggest("m3").is_empty());
        assert!(suggest("").is_empty());
        assert!(suggest(":/").is_empty());
    }

    #[test]
    fn an_oversized_request_suggests_nothing() {
        let requested = format!("BAAI/bge-m3{}", "x".repeat(MAX_REQUEST_LEN));
        assert!(suggest(&requested).is_empty());
    }

    #[test]
    fn suggestions_are_capped_and_ordered_closest_first() {
        // Three ids start with the request; the one with the fewest missing
        // characters comes first, and the order does not depend on input order.
        let expected = [
            "Qwen/Qwen3.5-4B",
            "Qwen/Qwen3.6-27B",
            "Qwen/Qwen3.8-27B-FP8",
        ];
        assert_eq!(suggest("Qwen/Qwen3"), expected);
        let mut reversed = catalog();
        reversed.reverse();
        assert_eq!(suggest_model_ids("Qwen/Qwen3", reversed), expected);

        let many = (0..10).map(|n| format!("org/model-{n}"));
        assert_eq!(suggest_model_ids("org/model", many).len(), MAX_SUGGESTIONS);
    }

    #[test]
    fn only_offered_candidates_are_ever_suggested() {
        // The caller's string is one edit from an id that is not in the
        // candidate set, so the hint cannot name it.
        let visible = vec!["BAAI/bge-m3".to_string()];
        assert!(suggest_model_ids("acme/secret-modle", visible).is_empty());
    }

    fn suggested(ids: &[&str]) -> NotFoundHint {
        NotFoundHint::Suggestions(ids.iter().map(|id| id.to_string()).collect())
    }

    #[test]
    fn the_message_keeps_the_plain_form_without_suggestions() {
        assert_eq!(
            unknown_model_message("x/y", &suggested(&[])),
            "Model 'x/y' not found"
        );
    }

    #[test]
    fn the_message_names_one_or_several_suggestions() {
        assert_eq!(
            unknown_model_message("bge-m3", &suggested(&["BAAI/bge-m3"])),
            "Model 'bge-m3' not found. Did you mean 'BAAI/bge-m3'?"
        );
        assert_eq!(
            unknown_model_message("q", &suggested(&["a/b", "c/d"])),
            "Model 'q' not found. Did you mean one of: 'a/b', 'c/d'?"
        );
    }

    #[test]
    fn the_retired_message_echoes_the_request_and_names_the_replacement() {
        let hint = NotFoundHint::Retired {
            replacement: "Qwen/Qwen3.6-27B".to_string(),
        };
        assert_eq!(
            unknown_model_message("qwen__qwen3-4b:fast", &hint),
            "Model 'qwen__qwen3-4b:fast' was retired. Use 'Qwen/Qwen3.6-27B'."
        );
    }

    fn retired_map(entries: &[(&str, &str)]) -> HashMap<String, RetiredModel> {
        crate::config::build_retired_models(
            entries
                .iter()
                .map(|(id, replacement)| (id.to_string(), replacement.to_string()))
                .collect(),
        )
    }

    fn retired() -> HashMap<String, RetiredModel> {
        retired_map(&[("Qwen/Qwen3-4B", "Qwen/Qwen3.6-27B")])
    }

    fn hint(requested: &str) -> NotFoundHint {
        hint_for(requested, &retired(), catalog(), |_| true)
    }

    fn retired_hint() -> NotFoundHint {
        NotFoundHint::Retired {
            replacement: "Qwen/Qwen3.6-27B".to_string(),
        }
    }

    #[test]
    fn a_retired_id_names_its_replacement() {
        assert_eq!(hint("Qwen/Qwen3-4B"), retired_hint());
    }

    #[test]
    fn a_retired_id_matches_in_any_case_and_in_the_double_underscore_form() {
        assert_eq!(hint("qwen/qwen3-4b"), retired_hint());
        assert_eq!(hint("Qwen__Qwen3-4B"), retired_hint());
    }

    #[test]
    fn a_retired_id_matches_with_a_bundle_prefix_or_a_profile_suffix() {
        assert_eq!(hint("default:/Qwen/Qwen3-4B"), retired_hint());
        assert_eq!(hint("Qwen/Qwen3-4B:fast"), retired_hint());
    }

    #[test]
    fn a_retired_id_is_matched_exactly_not_by_closeness() {
        // One edit away, and a different separator: both fall back to the
        // "Did you mean" ids.
        assert_eq!(hint("Qwen/Qwen3-4C"), suggested(&["Qwen/Qwen3.5-4B"]));
        assert!(matches!(
            hint("Qwen/Qwen3_4B"),
            NotFoundHint::Suggestions(_)
        ));
    }

    #[test]
    fn without_a_retired_entry_the_suggestions_are_unchanged() {
        assert_eq!(hint("bge-m3"), suggested(&["BAAI/bge-m3"]));
        assert_eq!(
            hint_for("bge-m3", &HashMap::new(), catalog(), |_| true),
            suggested(&["BAAI/bge-m3"])
        );
    }

    #[test]
    fn a_replacement_missing_from_the_registry_gives_no_retired_note() {
        let listed = vec!["Qwen/Qwen3.5-4B".to_string()];
        assert_eq!(
            hint_for("Qwen/Qwen3-4B", &retired(), listed, |_| true),
            suggested(&["Qwen/Qwen3.5-4B"])
        );
    }

    #[test]
    fn a_retired_or_replacement_id_the_caller_may_not_list_gives_no_retired_note() {
        let hides_replacement = |id: &str| id != "Qwen/Qwen3.6-27B";
        let hides_retired = |id: &str| id != "Qwen/Qwen3-4B";
        for visible in [&hides_replacement as &dyn Fn(&str) -> bool, &hides_retired] {
            let hint = hint_for("Qwen/Qwen3-4B", &retired(), catalog(), visible);
            assert!(matches!(hint, NotFoundHint::Suggestions(_)), "{hint:?}");
        }
    }

    #[test]
    fn the_policy_must_allow_the_retired_id_in_the_operators_case_and_in_lower_case() {
        let retired = retired_map(&[("bundle:/Org-42__Old:fast", "Qwen/Qwen3.6-27B")]);
        let asked = std::cell::RefCell::new(Vec::new());
        let records = |id: &str| {
            asked.borrow_mut().push(id.to_string());
            true
        };
        assert_eq!(
            hint_for("org-42/old", &retired, catalog(), records),
            retired_hint()
        );
        assert!(asked.borrow().iter().any(|id| id == "Org-42/Old"));
        assert!(asked.borrow().iter().any(|id| id == "org-42/old"));
        assert!(!asked.borrow().iter().any(|id| id == "Org-42__Old"));

        // Hiding either asked form blocks the note.
        for hidden in ["Org-42/Old", "org-42/old"] {
            let hint = hint_for("org-42/old", &retired, catalog(), |id: &str| id != hidden);
            assert!(
                matches!(hint, NotFoundHint::Suggestions(_)),
                "{hidden}: {hint:?}"
            );
        }
        // The `__` spelling is never asked about, so hiding it changes nothing.
        assert_eq!(
            hint_for("org-42/old", &retired, catalog(), |id: &str| id
                != "Org-42__Old"),
            retired_hint()
        );
    }

    #[test]
    fn a_lower_case_declared_id_is_asked_about_once() {
        let retired = retired_map(&[("org/old", "Qwen/Qwen3.6-27B")]);
        let asked = std::cell::RefCell::new(Vec::new());
        let records = |id: &str| {
            asked.borrow_mut().push(id.to_string());
            true
        };
        assert_eq!(
            hint_for("Org/Old", &retired, catalog(), records),
            retired_hint()
        );
        assert_eq!(
            asked.borrow().iter().filter(|id| *id == "org/old").count(),
            1
        );
    }

    #[test]
    fn a_replacement_written_in_another_case_or_spelling_names_the_registered_id() {
        for written in ["qwen/qwen3.6-27b", "QWEN__Qwen3.6-27B"] {
            let retired = retired_map(&[("Qwen/Qwen3-4B", written)]);
            let asked = std::cell::RefCell::new(Vec::new());
            let visible = |id: &str| {
                asked.borrow_mut().push(id.to_string());
                true
            };
            assert_eq!(
                hint_for("Qwen/Qwen3-4B", &retired, catalog(), visible),
                retired_hint(),
                "{written}"
            );
            assert!(asked.borrow().iter().any(|id| id == "Qwen/Qwen3.6-27B"));
            assert!(!asked.borrow().iter().any(|id| id == written));
            assert_eq!(
                unknown_model_message("Qwen/Qwen3-4B", &retired_hint()),
                "Model 'Qwen/Qwen3-4B' was retired. Use 'Qwen/Qwen3.6-27B'."
            );
        }
    }

    #[test]
    fn a_replacement_matching_several_registered_ids_names_the_byte_order_first() {
        let retired = retired_map(&[("Org/Old", "org/new")]);
        let listed = vec![
            "org/new".to_string(),
            "Org/New".to_string(),
            "Org/New:fast".to_string(),
        ];
        assert_eq!(
            hint_for("Org/Old", &retired, listed, |_| true),
            NotFoundHint::Retired {
                replacement: "Org/New".to_string()
            }
        );
    }

    #[test]
    fn a_variant_replacement_matches_only_that_registered_variant() {
        let listed = vec!["Org/New:fast".to_string()];
        assert_eq!(
            hint_for(
                "Org/Old",
                &retired_map(&[("Org/Old", "org/new:fast")]),
                listed.clone(),
                |_| true
            ),
            NotFoundHint::Retired {
                replacement: "Org/New:fast".to_string()
            }
        );
        let hint = hint_for(
            "Org/Old",
            &retired_map(&[("Org/Old", "Org/New")]),
            listed,
            |_| true,
        );
        assert!(matches!(hint, NotFoundHint::Suggestions(_)), "{hint:?}");
    }

    #[test]
    fn an_oversized_request_gets_no_retired_note() {
        let long = format!("{}{}", "x".repeat(MAX_REQUEST_LEN), "/model");
        let retired = retired_map(&[(&long, "Qwen/Qwen3.6-27B")]);
        assert_eq!(
            hint_for(&long, &retired, catalog(), |_| true),
            suggested(&[])
        );
    }

    #[test]
    fn edit_distance_counts_insertions_deletions_and_substitutions() {
        assert_eq!(edit_distance(b"", b""), 0);
        assert_eq!(edit_distance(b"abc", b""), 3);
        assert_eq!(edit_distance(b"kitten", b"sitting"), 3);
        assert_eq!(edit_distance(b"bgem3", b"bgen3"), 1);
    }
}
