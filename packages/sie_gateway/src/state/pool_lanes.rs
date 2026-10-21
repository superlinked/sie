//! Which physical lanes a logical pool's pinned models need.
//!
//! A pinned model is served on the lane of the bundle routing resolves it to,
//! or on a composite lane that runs that bundle (see
//! [`PhysicalLaneCatalog::serving_bundle`]). A pool whose pins span bundles
//! therefore needs several lanes per machine profile, and its warm floor must
//! keep each of them warm; a pin that no lane of the pool can serve must be
//! refused when the pool is created rather than silently never loaded.

use std::collections::BTreeSet;

use crate::state::demand_tracker::PhysicalLaneCatalog;
use crate::state::model_registry::ModelRegistry;

/// The routing bundle of every pin that resolves to one.
fn pin_bundles(registry: &ModelRegistry, pinned_models: &[String]) -> Vec<String> {
    pinned_models
        .iter()
        .filter_map(|model| registry.resolve_bundle(model, None).ok())
        .collect()
}

/// The lane bundles a pool keeps warm on `machine_profile`.
///
/// A pool restricted to a bundle keeps that bundle's lane warm. Otherwise each
/// pinned model's lane is kept warm, and a pool without pins keeps the
/// `default` bundle's lane warm, as before pins existed. Each bundle maps to
/// the lane that serves it on this profile, which is a composite lane where
/// the bundle has none of its own.
pub fn warm_lane_bundles(
    registry: &ModelRegistry,
    catalog: &PhysicalLaneCatalog,
    queue_pool: &str,
    machine_profile: &str,
    bundle_filter: Option<&str>,
    pinned_models: &[String],
) -> Vec<String> {
    let logical = match bundle_filter {
        Some(bundle) => vec![bundle.to_string()],
        None => {
            let pins = pin_bundles(registry, pinned_models);
            if pins.is_empty() {
                vec!["default".to_string()]
            } else {
                pins
            }
        }
    };
    let lanes: BTreeSet<String> = logical
        .iter()
        .map(|bundle| {
            catalog
                .serving_bundle(
                    queue_pool,
                    &[machine_profile.to_string()],
                    bundle,
                    &registry.composites_covering(bundle),
                )
                .trim()
                .to_ascii_lowercase()
        })
        .collect();
    lanes.into_iter().collect()
}

/// Explain each pinned model that no lane of the pool can serve.
///
/// With a `bundle_filter` the pool only takes workers of that bundle, so each
/// pin must resolve to that bundle or to a component of it. Every pin must
/// also have a configured lane on at least one of the pool's machine profiles.
/// An empty catalog (a deployment without a Helm lane catalog) skips the lane
/// check.
pub fn unservable_pins(
    registry: &ModelRegistry,
    catalog: &PhysicalLaneCatalog,
    queue_pool: &str,
    machine_profiles: &[String],
    bundle_filter: Option<&str>,
    pinned_models: &[String],
) -> Vec<String> {
    let mut problems = Vec::new();
    for model in pinned_models {
        let bundle = match registry.resolve_bundle(model, None) {
            Ok(bundle) => bundle,
            Err(error) => {
                problems.push(format!(
                    "pinned model '{model}' has no routable bundle ({error})"
                ));
                continue;
            }
        };
        if let Some(filter) = bundle_filter {
            if !registry.lane_serves_bundle(filter, &bundle) {
                problems.push(format!(
                    "pinned model '{model}' runs on bundle '{bundle}', but the pool only takes '{filter}' workers"
                ));
                continue;
            }
        }
        if catalog.is_empty() {
            continue;
        }
        let composites = registry.composites_covering(&bundle);
        let served = machine_profiles.iter().any(|profile| {
            let lane = match bundle_filter {
                Some(filter) => filter.to_string(),
                None => catalog.serving_bundle(
                    queue_pool,
                    std::slice::from_ref(profile),
                    &bundle,
                    &composites,
                ),
            };
            catalog.resolve(queue_pool, profile, &lane).is_some()
        });
        if !served {
            let hint = if composites.is_empty() {
                String::new()
            } else {
                format!(
                    " or a composite bundle that runs it ({})",
                    composites.join(", ")
                )
            };
            problems.push(format!(
                "pinned model '{model}' runs on bundle '{bundle}', which has no worker lane on queue pool \
                 '{queue_pool}' for machine profile(s) {}; configure that bundle{hint} on one of them",
                machine_profiles.join(", ")
            ));
        }
    }
    problems
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use tempfile::TempDir;

    fn write(dir: &Path, name: &str, body: &str) {
        std::fs::write(dir.join(name), body).unwrap();
    }

    /// A registry with an encoder (default), an LLM (sglang) and the
    /// `default-sglang` composite that runs both.
    fn registry() -> (TempDir, ModelRegistry) {
        let temp = TempDir::new().unwrap();
        let bundles = temp.path().join("bundles");
        let models = temp.path().join("models");
        std::fs::create_dir_all(&bundles).unwrap();
        std::fs::create_dir_all(&models).unwrap();
        write(
            &bundles,
            "default.yaml",
            "name: default\npriority: 10\nadapters:\n  - sie_server.adapters.sentence_transformer\n",
        );
        write(
            &bundles,
            "sglang.yaml",
            "name: sglang\npriority: 20\nadapters:\n  - sie_server.adapters.sglang.generation\n",
        );
        write(
            &bundles,
            "default-sglang.yaml",
            "name: default-sglang\npriority: 90\ncomponents: [default, sglang]\nadapters:\n  \
             - sie_server.adapters.sentence_transformer\n  - sie_server.adapters.sglang.generation\n",
        );
        write(
            &models,
            "enc.yaml",
            "sie_id: acme/enc\nhf_id: acme/enc\ntasks:\n  encode:\n    dense:\n      dim: 8\nprofiles:\n  default:\n    \
             adapter_path: sie_server.adapters.sentence_transformer:A\n    max_batch_tokens: 8\n",
        );
        write(
            &models,
            "llm.yaml",
            "sie_id: acme/llm\nhf_id: acme/llm\ntasks:\n  generate:\n    context_length: 8\n    max_output_tokens: 8\nprofiles:\n  default:\n    \
             adapter_path: sie_server.adapters.sglang.generation:G\n    max_batch_tokens: 8\n",
        );
        let registry = ModelRegistry::new(&bundles, &models, true);
        (temp, registry)
    }

    fn catalog(lanes: &[(&str, &str, &str)]) -> PhysicalLaneCatalog {
        PhysicalLaneCatalog::try_from_raw(lanes.iter().map(|(pool, profile, bundle)| {
            (pool.to_string(), profile.to_string(), bundle.to_string())
        }))
        .unwrap()
    }

    fn pins() -> Vec<String> {
        vec!["acme/llm".to_string(), "acme/enc".to_string()]
    }

    #[test]
    fn composite_lane_serves_both_pins_on_its_profile() {
        let (_temp, registry) = registry();
        let catalog = catalog(&[
            ("default", "h100", "default-sglang"),
            ("default", "l4", "default"),
        ]);

        assert_eq!(
            warm_lane_bundles(&registry, &catalog, "default", "h100", None, &pins()),
            vec!["default-sglang"]
        );
        assert!(unservable_pins(
            &registry,
            &catalog,
            "default",
            &["h100".into()],
            None,
            &pins()
        )
        .is_empty());
    }

    #[test]
    fn pins_across_bundles_keep_every_lane_warm() {
        let (_temp, registry) = registry();
        let catalog = catalog(&[
            ("default", "h100", "default"),
            ("default", "h100", "sglang"),
        ]);

        assert_eq!(
            warm_lane_bundles(&registry, &catalog, "default", "h100", None, &pins()),
            vec!["default", "sglang"]
        );
        // Without pins the pool keeps the default lane warm, as before.
        assert_eq!(
            warm_lane_bundles(&registry, &catalog, "default", "h100", None, &[]),
            vec!["default"]
        );
    }

    #[test]
    fn a_pin_without_a_lane_is_reported() {
        let (_temp, registry) = registry();
        let catalog = catalog(&[("default", "l4", "default")]);

        let problems = unservable_pins(
            &registry,
            &catalog,
            "default",
            &["l4".into()],
            None,
            &pins(),
        );

        assert_eq!(problems.len(), 1);
        assert!(problems[0].contains("'acme/llm'"), "{problems:?}");
        assert!(problems[0].contains("default-sglang"), "{problems:?}");
    }

    #[test]
    fn a_bundle_filter_must_run_every_pin() {
        let (_temp, registry) = registry();
        let catalog = catalog(&[
            ("default", "h100", "default-sglang"),
            ("default", "h100", "default"),
        ]);

        let only_default = unservable_pins(
            &registry,
            &catalog,
            "default",
            &["h100".into()],
            Some("default"),
            &pins(),
        );
        assert_eq!(only_default.len(), 1);
        assert!(
            only_default[0].contains("only takes 'default' workers"),
            "{only_default:?}"
        );

        assert!(unservable_pins(
            &registry,
            &catalog,
            "default",
            &["h100".into()],
            Some("default-sglang"),
            &pins()
        )
        .is_empty());
    }

    #[test]
    fn without_a_lane_catalog_only_bundles_are_checked() {
        let (_temp, registry) = registry();
        let empty = PhysicalLaneCatalog::default();

        assert!(unservable_pins(
            &registry,
            &empty,
            "default",
            &["h100".into()],
            None,
            &pins()
        )
        .is_empty());
        assert_eq!(
            unservable_pins(
                &registry,
                &empty,
                "default",
                &["h100".into()],
                None,
                &["nope/x".into()]
            )
            .len(),
            1
        );
    }
}
