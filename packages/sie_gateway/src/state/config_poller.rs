//! Periodic staleness reconciliation against `sie-config`.
//!
//! The gateway receives live config deltas over NATS Core pub/sub. Core
//! pub/sub has no replay: if a delta is published while the gateway is
//! disconnected (or lands between reconnect and subscription resume), it is
//! gone for this replica. NATS connection events alone cannot detect this
//! class of drift.
//!
//! This poller closes that gap by periodically calling
//! `GET /v1/configs/epoch` on `sie-config` and comparing the response to
//! the gateway's last-known control-plane state on TWO axes:
//!
//! 1. **Model epoch** (`ConfigEpoch`, monotonic counter bumped on every
//!    model-config write). A remote-ahead value triggers a full
//!    `/v1/configs/export` fetch to catch up on missed writes.
//! 2. **Bundle hash** (`BundlesHash`, sha256 fingerprint of `sie-config`'s
//!    loaded bundle set). Bundles are filesystem artifacts inside the
//!    `sie-config` image; their "version" is effectively redeploy time,
//!    which the model-write counter does not observe. A mismatch triggers
//!    a full bootstrap so the gateway picks up bundle additions or changes
//!    without requiring a gateway pod restart (and without depending on a
//!    coincidental model write to bump the epoch).
//!
//! Both axes funnel into the same `bootstrap_once` call — that function
//! re-fetches bundles and models together, which is the correct blanket
//! response to any control-plane drift we observe.
//!
//! Design notes:
//!
//! - The poller always compares local vs remote, even while the local epoch
//!   is still 0. That matters for fresh clusters: `sie-config` starts at
//!   epoch 0 too, so a successful first-ever bootstrap leaves the local
//!   counter at 0. If the very first write on `sie-config` (→ epoch 1)
//!   arrives over NATS and is lost in flight, we need the poller to catch
//!   remote=1 vs local=0 and trigger a catch-up. Gating on `local != 0`
//!   would wedge the gateway silently until the next restart.
//! - The epoch endpoint is a deliberately tiny payload (one integer +
//!   one short hex string). It is much cheaper than `/export` and can run
//!   on a short cadence without stressing `sie-config`.
//! - Errors are logged and swallowed. A transient epoch-endpoint failure
//!   does not reset the local epoch or the stored bundle hash; the next
//!   tick retries.
//! - `local > remote` recovery on the epoch axis: if the local counter has
//!   run ahead of authority (buggy publisher, forged delta, or — most
//!   likely — `sie-config` lost its epoch file and restarted at 0), the
//!   poller re-runs `bootstrap_once` and, on success, force-resets the
//!   local counter to the remote value. Without this branch the
//!   `remote > local` catch-up trigger would be permanently unreachable.
//! - Empty remote `bundles_hash` means `sie-config` is in a
//!   registry-degraded state (startup failure or hot reload in flight).
//!   We skip the hash-mismatch branch in that case rather than thrash
//!   fetching an empty bundle set.

use std::sync::Arc;
use std::time::Duration;

use tracing::{debug, error, info, warn};

use crate::state::bundles_hash::BundlesHash;
use crate::state::config_bootstrap::{bootstrap_once, BootstrapClient, BootstrapError};
use crate::state::config_epoch::ConfigEpoch;
use crate::state::model_registry::ModelRegistry;

pub const DEFAULT_POLL_INTERVAL: Duration = Duration::from_secs(30);

/// Spawn the periodic epoch-poll reconciliation loop.
///
/// Returns `None` if `base_url` is unset (no control plane configured).
/// Otherwise returns a handle to the running task.
pub fn spawn(
    base_url: Option<&str>,
    admin_token: Option<&str>,
    registry: Arc<ModelRegistry>,
    config_epoch: ConfigEpoch,
    bundles_hash: BundlesHash,
    interval: Duration,
) -> Option<tokio::task::JoinHandle<()>> {
    let base = base_url?.to_string();
    let admin_token = admin_token.map(str::to_string);

    let handle = tokio::spawn(async move {
        let client = match BootstrapClient::new(base, admin_token) {
            Ok(c) => c,
            Err(e) => {
                warn!(error = %e, "failed to build epoch-poll client; giving up");
                // Same reasoning as `config_bootstrap.rs`: a broken
                // HTTP client config prevents the epoch poller from
                // ever running, which looks identical on dashboards
                // to a sie-config outage. Count it once.
                crate::metrics::CONFIG_BOOTSTRAP_FAILURES.inc();
                return;
            }
        };
        let mut ticker = tokio::time::interval(interval);
        // Skip the immediate first tick: bootstrap just ran (or is retrying).
        // No need to hammer /epoch before at least one interval has passed.
        ticker.tick().await;

        loop {
            ticker.tick().await;
            let local_epoch_val = config_epoch.get();
            let local_bundles_hash = bundles_hash.get();
            let snapshot = match client.fetch_epoch().await {
                Ok(s) => s,
                Err(BootstrapError::BadStatus { status, .. }) => {
                    warn!(status = status, "epoch poll returned non-success");
                    continue;
                }
                Err(e) => {
                    warn!(error = %e, "epoch poll failed");
                    continue;
                }
            };

            let remote_epoch = snapshot.epoch;
            let remote_bundles_hash = snapshot.bundles_hash;
            // Skip the bundle-hash comparison when sie-config reported the
            // empty "registry unavailable" sentinel. Treating empty as a
            // real change would make us thrash every tick while the control
            // plane is booting or hot-reloading. The epoch-drift branch
            // still runs normally.
            let bundles_hash_drift =
                !remote_bundles_hash.is_empty() && remote_bundles_hash != local_bundles_hash;

            // Branch priority matters when both axes drift in the same
            // tick (e.g. sie-config restarts with an ephemeral store: the
            // epoch rewinds AND the bundles_hash shifts because the
            // restarted instance recomputed it from the baked baseline).
            // The `remote_epoch < local_epoch` branch is the only one
            // that calls `force_set`. If we handled `bundles_hash_drift`
            // first in the combined case, `bootstrap_once` would run
            // `set_max` (a no-op while local > remote), skip the
            // `force_set`, and leave the gateway one 30-s tick ahead of
            // authority until the next poll. Checking the recovery path
            // first collapses the combined case into a single bootstrap
            // that both reinstalls bundles and rewinds the counter.
            if remote_epoch < local_epoch_val {
                // Local counter ran ahead of authority. Most innocent
                // cause: sie-config lost its persisted epoch file and
                // restarted at a lower value. Most worrying cause: a
                // forged NATS delta wedged the gateway ahead of authority
                // — without this branch the normal `remote > local`
                // catch-up trigger would be permanently unreachable.
                //
                // Order matters for retry semantics: fetch the
                // authoritative export FIRST, then force_set the local
                // counter only on success. If we force_set first and the
                // export fetch then fails, the next tick sees local ==
                // remote and logs "in sync" forever — recovery wedges
                // silently. With the order below, a failed export keeps
                // local > remote, so the next tick re-enters this branch
                // and retries.
                //
                // Note: ModelRegistry is append-only. A successful
                // bootstrap overlays authority on top of any profiles
                // admitted during the pre-recovery window; it does not
                // remove them. A restart is the clean fix if the ghost
                // entries matter.
                //
                // Logged at ERROR because neither cause is normal
                // operation.
                error!(
                    local_epoch = local_epoch_val,
                    remote_epoch = remote_epoch,
                    bundles_hash_drift,
                    "local epoch is ahead of sie-config authority; re-fetching export before force-reset"
                );
                match bootstrap_once(&client, registry.as_ref(), &config_epoch, &bundles_hash).await
                {
                    Ok(_) => {
                        // `bootstrap_once` advanced via `set_max`, which is
                        // a no-op while local > remote. Unconditionally
                        // reset now that the registry is in a known state.
                        // This path ALSO handles a coincident
                        // `bundles_hash_drift` because `bootstrap_once`
                        // already refreshed bundles via `install_bundles`
                        // and stored the new hash inside `BundlesHash`.
                        config_epoch.force_set(remote_epoch);
                    }
                    Err(e) => {
                        warn!(
                            error = %e,
                            "recovery export fetch failed; keeping stale local epoch and retrying next tick"
                        );
                    }
                }
            } else if remote_epoch > local_epoch_val || bundles_hash_drift {
                info!(
                    local_epoch = local_epoch_val,
                    remote_epoch = remote_epoch,
                    bundles_hash_drift,
                    reason = if remote_epoch > local_epoch_val {
                        "epoch ahead"
                    } else {
                        "bundle hash changed"
                    },
                    "control-plane drift detected; re-running bootstrap"
                );
                if let Err(e) =
                    bootstrap_once(&client, registry.as_ref(), &config_epoch, &bundles_hash).await
                {
                    warn!(error = %e, "catch-up bootstrap failed; will retry next tick");
                }
            } else {
                debug!(
                    local_epoch = local_epoch_val,
                    remote_epoch = remote_epoch,
                    "epoch poll: in sync"
                );
            }
        }
    });

    Some(handle)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;
    use tempfile::TempDir;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use crate::types::bundle::BundleInfo;

    /// Empty filesystem registry with a pre-installed `default` bundle — the
    /// same shape `BootstrapClient::bootstrap` produces after the bundle
    /// fetch. Kept independent of the `config_bootstrap` test helper so
    /// these tests don't drift if that helper changes.
    fn make_registry() -> (Arc<ModelRegistry>, TempDir) {
        let temp = TempDir::new().unwrap();
        let bundles = temp.path().join("bundles");
        let models = temp.path().join("models");
        std::fs::create_dir_all(&bundles).unwrap();
        std::fs::create_dir_all(&models).unwrap();
        let registry = Arc::new(ModelRegistry::new(&bundles, &models, true));
        registry.install_bundles(vec![BundleInfo {
            name: "default".to_string(),
            priority: 10,
            adapters: vec!["sie_server.adapters.sentence_transformer".to_string()],
        }]);
        (registry, temp)
    }

    async fn mount_bundles(server: &MockServer) {
        Mock::given(method("GET"))
            .and(path("/v1/configs/bundles"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "bundles": [{
                    "bundle_id": "default",
                    "priority": 10,
                    "adapter_count": 1,
                    "source": "filesystem",
                }],
            })))
            .mount(server)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/configs/bundles/default"))
            .respond_with(ResponseTemplate::new(200).set_body_string(
                "name: default\npriority: 10\nadapters:\n  - sie_server.adapters.sentence_transformer\n",
            ))
            .mount(server)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/configs/export"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "snapshot_version": 1,
                "epoch": 0,
                "generated_at": "2026-04-17T00:00:00Z",
                "models": [],
            })))
            .mount(server)
            .await;
    }

    /// The load-bearing test for the whole point of the bundles_hash
    /// plumbing: when the remote hash differs from the stored hash AND the
    /// epoch is NOT ahead, the poller must still trigger a full bootstrap.
    /// Without this, a sie-config redeploy that adds a bundle would not
    /// propagate to the gateway until the next model write bumped the
    /// epoch — exactly the operational gap that motivates this change.
    #[tokio::test]
    async fn bundles_hash_drift_triggers_bootstrap_without_epoch_advance() {
        let server = MockServer::start().await;
        mount_bundles(&server).await;
        Mock::given(method("GET"))
            .and(path("/v1/configs/epoch"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "epoch": 0,
                "bundles_hash": "new_hash_after_cfg_redeploy",
            })))
            .mount(&server)
            .await;

        let (registry, _tmp) = make_registry();
        let config_epoch = ConfigEpoch::new();
        let bundles_hash = BundlesHash::new();
        // Pretend a prior bootstrap recorded a different hash — the
        // "before the sie-config redeploy" state.
        bundles_hash.store("stale_hash".to_string());

        let handle = spawn(
            Some(&server.uri()),
            None,
            Arc::clone(&registry),
            config_epoch.clone(),
            bundles_hash.clone(),
            Duration::from_millis(50),
        )
        .expect("poller should spawn");

        // Wait for the poller to observe drift and re-run bootstrap. The
        // first tick is skipped by design, so we wait several intervals.
        tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                if bundles_hash.get() == "new_hash_after_cfg_redeploy" {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("poller should pick up bundle-hash drift within 3s");

        assert_eq!(bundles_hash.get(), "new_hash_after_cfg_redeploy");
        // Epoch path was NOT the trigger — remote == local == 0 — proving
        // the hash-drift branch stands on its own.
        assert_eq!(config_epoch.get(), 0);

        handle.abort();
    }

    /// Defense-in-depth for the ephemeral-store `sie-config` restart
    /// case: if `sie-config` is deployed with `config.configStore.enabled=false`
    /// (the Helm default), its persisted epoch file is gone after every
    /// pod restart and the counter comes back at 0. The normal
    /// `remote_epoch > local_epoch` trigger would stay permanently dark
    /// because local has run ahead, so without this branch the gateway
    /// would keep whatever stale catalog it had at the moment of the
    /// restart. This test drives the full poller loop through the
    /// recovery path: remote=3 vs local=10, `bootstrap_once` runs, and
    /// `force_set` rewinds the gateway's counter to the authority's
    /// value. Also guards against a forged NATS delta artificially
    /// advancing the gateway — the same recovery applies.
    #[tokio::test]
    async fn remote_epoch_behind_local_triggers_force_reset_after_bootstrap() {
        let server = MockServer::start().await;
        mount_bundles(&server).await;
        Mock::given(method("GET"))
            .and(path("/v1/configs/epoch"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "epoch": 3,
                "bundles_hash": "post_restart_hash",
            })))
            .mount(&server)
            .await;

        let (registry, _tmp) = make_registry();
        let config_epoch = ConfigEpoch::new();
        // Pretend earlier deltas (or a forged one) advanced the gateway
        // well past what sie-config's post-restart `/epoch` now reports.
        config_epoch.set_max(10);
        assert_eq!(config_epoch.get(), 10);
        let bundles_hash = BundlesHash::new();
        bundles_hash.store("pre_restart_hash".to_string());

        let handle = spawn(
            Some(&server.uri()),
            None,
            Arc::clone(&registry),
            config_epoch.clone(),
            bundles_hash.clone(),
            Duration::from_millis(50),
        )
        .expect("poller should spawn");

        // Wait for force_set to rewind the local counter onto authority.
        // Using the counter — not the hash — as the observation point:
        // set_max cannot pull local *down*, so a successful recovery is
        // the only path that can produce local == 3.
        tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                if config_epoch.get() == 3 {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("poller should force-reset local epoch to remote within 3s");

        // Bootstrap actually ran before the force_set (docs §4.2: export
        // first, reset only on success). The recorded hash proves we
        // took the recovery path, not just a hash-drift tick — though
        // either would be acceptable here since both converge on the
        // right end state.
        assert_eq!(config_epoch.get(), 3);
        assert_eq!(bundles_hash.get(), "post_restart_hash");

        handle.abort();
    }

    /// Regression: when `remote_epoch < local_epoch` AND `bundles_hash`
    /// also drifts in the same tick (ephemeral `sie-config` restart),
    /// the recovery branch must fire first so the single bootstrap that
    /// refreshes bundles also rewinds the counter. The old ordering
    /// took the hash-drift branch first, which ran `bootstrap_once`
    /// (`set_max` is a no-op while local > remote), skipped `force_set`,
    /// and required a SECOND tick (hash now equal, `remote < local`
    /// branch) to finally `force_set`. In prod that's a 30 s window of
    /// known-stale counter PLUS a duplicate `bootstrap_once` — two
    /// `/v1/configs/export` fetches instead of one. The `expect(1)`
    /// mock on `/export` locks the single-bootstrap contract in.
    #[tokio::test]
    async fn combined_epoch_rewind_and_bundle_drift_converges_in_one_bootstrap() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/configs/epoch"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "epoch": 3,
                "bundles_hash": "post_restart_hash",
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/configs/bundles"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "bundles": [{
                    "bundle_id": "default",
                    "priority": 10,
                    "adapter_count": 1,
                    "source": "filesystem",
                }],
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/configs/bundles/default"))
            .respond_with(ResponseTemplate::new(200).set_body_string(
                "name: default\npriority: 10\nadapters:\n  - sie_server.adapters.sentence_transformer\n",
            ))
            .mount(&server)
            .await;
        // Exactly-once expectation: old ordering would fire this twice
        // (hash-drift bootstrap on tick 1, recovery bootstrap on tick 2).
        Mock::given(method("GET"))
            .and(path("/v1/configs/export"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "snapshot_version": 1,
                "epoch": 3,
                "generated_at": "2026-04-17T00:00:00Z",
                "models": [],
            })))
            .expect(1)
            .mount(&server)
            .await;

        let (registry, _tmp) = make_registry();
        let config_epoch = ConfigEpoch::new();
        config_epoch.set_max(10);
        let bundles_hash = BundlesHash::new();
        bundles_hash.store("pre_restart_hash".to_string());

        let interval = Duration::from_millis(50);
        let handle = spawn(
            Some(&server.uri()),
            None,
            Arc::clone(&registry),
            config_epoch.clone(),
            bundles_hash.clone(),
            interval,
        )
        .expect("poller should spawn");

        // Wait for the recovery branch to converge. `config_epoch == 3`
        // is the single observable that unambiguously says "force_set
        // happened": `set_max` cannot pull local down, so the only path
        // to local == 3 is the recovery branch. Poll-based wait keeps
        // the test fast on quick runners and tolerant on slow CI.
        tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                if config_epoch.get() == 3 {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("poller should converge within 3s");
        assert_eq!(bundles_hash.get(), "post_restart_hash");

        // Give the poller a few more ticks to prove we do NOT thrash
        // (the expect(1) matcher fails verification if a second
        // bootstrap sneaks in while local == remote).
        tokio::time::sleep(interval * 4).await;
        handle.abort();
        server.verify().await;
    }

    /// Empty remote hash is the "sie-config registry unavailable" sentinel.
    /// The poller must treat it as "nothing to sync" and keep the stored
    /// hash intact rather than thrash against a degraded control plane.
    #[tokio::test]
    async fn empty_remote_bundles_hash_does_not_trigger_bootstrap() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/configs/epoch"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "epoch": 0,
                "bundles_hash": "",
            })))
            .mount(&server)
            .await;
        // Lock in "no thrash" by installing expect(0) mocks on the
        // bootstrap endpoints. If the poller mistakenly enters the
        // re-fetch branch these matchers tick their counters and the
        // MockServer destructor fails the test — stronger than just
        // asserting local state is unchanged, which is ambiguous with a
        // failed bootstrap leaving the pre-state intact.
        Mock::given(method("GET"))
            .and(path("/v1/configs/bundles"))
            .respond_with(ResponseTemplate::new(500))
            .expect(0)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/configs/export"))
            .respond_with(ResponseTemplate::new(500))
            .expect(0)
            .mount(&server)
            .await;

        let (registry, _tmp) = make_registry();
        let config_epoch = ConfigEpoch::new();
        let bundles_hash = BundlesHash::new();
        bundles_hash.store("stale_hash".to_string());

        let handle = spawn(
            Some(&server.uri()),
            None,
            Arc::clone(&registry),
            config_epoch.clone(),
            bundles_hash.clone(),
            Duration::from_millis(50),
        )
        .expect("poller should spawn");

        tokio::time::sleep(Duration::from_millis(300)).await;
        assert_eq!(bundles_hash.get(), "stale_hash");

        handle.abort();
        server.verify().await;
    }
}
