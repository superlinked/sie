//! "Did you mean ...?" hints for the `MODEL_NOT_FOUND` response.
//!
//! The hint is a function of exactly two inputs: the string the caller sent
//! and the model ids that caller may list. Nothing resolved from the request
//! (an alias target, a canonical id, a governed route) takes part, so a model
//! hidden by a [`ModelAccessPolicy`](crate::server::ModelAccessPolicy), a model
//! that does not exist, and a model whose route was never compiled still
//! produce the same body for the same request string.

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

/// The `MODEL_NOT_FOUND` message for `requested`, with a hint when
/// `suggestions` is not empty.
///
/// `requested` is always the caller's own string.
pub(crate) fn unknown_model_message(requested: &str, suggestions: &[String]) -> String {
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

/// Suggestions for `requested`, drawn only from the ids this caller may list.
///
/// Candidates are the registry's ids that pass the visibility filter
/// `GET /v1/models` applies, so every suggestion is an id that listing shows.
/// The listing can show more: ids only a worker advertises are left out,
/// because a request for one gets this same 404 from a populated registry.
pub(crate) fn visible_model_suggestions(
    state: &AppState,
    ext: &axum::http::Extensions,
    requested: &str,
) -> Vec<String> {
    let candidates = state.model_registry.list_models().into_iter().filter(|id| {
        state
            .model_access_policy
            .as_ref()
            .is_none_or(|policy| policy.visible(id, ext))
    });
    suggest_model_ids(requested, candidates)
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

    #[test]
    fn the_message_keeps_the_plain_form_without_suggestions() {
        assert_eq!(unknown_model_message("x/y", &[]), "Model 'x/y' not found");
    }

    #[test]
    fn the_message_names_one_or_several_suggestions() {
        assert_eq!(
            unknown_model_message("bge-m3", &["BAAI/bge-m3".to_string()]),
            "Model 'bge-m3' not found. Did you mean 'BAAI/bge-m3'?"
        );
        assert_eq!(
            unknown_model_message("q", &["a/b".to_string(), "c/d".to_string()]),
            "Model 'q' not found. Did you mean one of: 'a/b', 'c/d'?"
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
