//! Per-model scheduler registry.
//!
//! The registry owns one [`Scheduler`] per `model_id` and hands it
//! back on demand:
//!
//! * [`SchedulerRegistry::get_or_create`] returns `(Arc<Scheduler<..>>, created)`.
//!   A `created == true` flag signals the dispatcher that this was
//!   the first call for the model and a per-model drain loop needs
//!   to be spawned now; subsequent calls return the same shared
//!   [`Arc`] with `created == false`, so the scheduler's adaptive
//!   controller state and per-LoRA batcher map persist across
//!   requests.
//! * Schedulers are lazily created — models that never receive
//!   traffic never allocate one.
//!
//! ## Per-model batch cost budget
//!
//! Each scheduler's static cost cap and adaptive cost range derive from
//! the model's own `max_batch_tokens`, which the backend reports with
//! `EnsureModelReady`, as Python's `ModelWorker` derives them from the
//! model's profile. Precedence:
//!
//! 1. `SIE_BATCHER_MAX_BATCH_COST`, when set: [`SchedulerRegistry::from_env`]
//!    builds the registry cost-pinned and every model keeps that cap,
//!    ignoring the reported budget.
//! 2. Otherwise the budget the backend reported for the model.
//! 3. Otherwise (a backend that does not report one) the default
//!    config's cap, 16384.
//!
//! `SIE_ADAPTIVE_BATCH_*` overrides still apply on top of whichever
//! budget wins (see
//! [`crate::scheduler::AdaptiveBatchController::from_batch_config_and_env`]),
//! and the static cost cap starts at the controller's starting cost, so an
//! overridden cost range holds from a scheduler's first batch.
//!
//! There is no per-model scheduler env list any more. Active models route
//! through the Rust scheduler when their worker pool is the sidecar pool.
//! Every model that lands on a worker-sidecar goes through the
//! scheduler.
//!
//! Kept generic over `<I: HasCost, T>` so this module is testable in
//! isolation against the concrete dispatcher item / work metadata
//! types.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;
use tracing::info;

use super::batch_config::{process_env, BatchConfig, EnvLookup};
use super::batch_former::HasCost;
use super::engine::Scheduler;

/// Per-model scheduler registry.
pub struct SchedulerRegistry<I: HasCost, T> {
    /// Default caps + per-scheduler defaults every lazily created
    /// [`Scheduler`] starts with. Its `max_batch_cost` is replaced by
    /// the model's reported batch cost budget unless the registry is
    /// cost-pinned.
    default_config: BatchConfig,
    /// `true` when the operator pinned the cost cap
    /// (`SIE_BATCHER_MAX_BATCH_COST`): reported model budgets are
    /// ignored and every scheduler keeps `default_config.max_batch_cost`.
    cost_pinned: bool,
    /// Where the schedulers read their `SIE_ADAPTIVE_BATCH_*` overrides.
    env: EnvLookup,
    schedulers: RwLock<HashMap<String, Arc<Scheduler<I, T>>>>,
}

impl<I, T> std::fmt::Debug for SchedulerRegistry<I, T>
where
    I: HasCost,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SchedulerRegistry")
            .field("default_config", &self.default_config)
            .field("cost_pinned", &self.cost_pinned)
            .finish_non_exhaustive()
    }
}

impl<I, T> SchedulerRegistry<I, T>
where
    I: HasCost + Send + Sync + 'static,
    T: Send + Sync + 'static,
{
    /// Build with the given default per-scheduler config. Model budgets
    /// reported to [`Self::get_or_create`] replace its cost cap.
    #[must_use]
    pub fn new(default_config: BatchConfig) -> Self {
        Self {
            default_config,
            cost_pinned: false,
            env: process_env(),
            schedulers: RwLock::new(HashMap::new()),
        }
    }

    /// Production constructor: `SIE_BATCHER_*` defaults, cost-pinned
    /// exactly when `SIE_BATCHER_MAX_BATCH_COST` is set.
    #[must_use]
    pub fn from_env() -> Self {
        Self::from_lookup(process_env())
    }

    /// [`Self::from_env`] reading variables through `env`, which the
    /// schedulers also use for their `SIE_ADAPTIVE_BATCH_*` overrides.
    #[must_use]
    pub fn from_lookup(env: EnvLookup) -> Self {
        Self {
            default_config: BatchConfig::from_lookup(&*env),
            cost_pinned: BatchConfig::max_batch_cost_override(&*env).is_some(),
            env,
            schedulers: RwLock::new(HashMap::new()),
        }
    }

    /// Pin every scheduler to `default_config.max_batch_cost`, ignoring
    /// the budgets models report. [`Self::from_env`] pins exactly when
    /// `SIE_BATCHER_MAX_BATCH_COST` is set.
    #[must_use]
    pub fn with_cost_pinned(mut self, cost_pinned: bool) -> Self {
        self.cost_pinned = cost_pinned;
        self
    }

    /// Return the scheduler for `model_id`, lazily creating it on
    /// first call.
    ///
    /// `model_max_batch_tokens` is the model's batch cost budget as the
    /// backend reported it (`None` when it did not; zero counts as
    /// absent). A new scheduler starts from it; an existing one re-bases
    /// onto it when it differs from the budget the scheduler runs on (see
    /// [`Scheduler::adopt_cost_budget`]). `None` never resets a budget a
    /// scheduler already adopted. A cost-pinned registry ignores it.
    ///
    /// The second tuple element is `true` only on the call that
    /// actually materialised the scheduler. The dispatcher uses it
    /// to spawn exactly one drain loop per model — see
    /// `crate::dispatcher::Dispatcher::resolve_scheduler`.
    ///
    /// Thread-safe: the hot path is a read-lock + hashmap lookup plus
    /// one atomic load; only the very first call for a given `model_id`
    /// takes the write lock, and concurrent callers double-check under it.
    pub async fn get_or_create(
        &self,
        model_id: &str,
        model_max_batch_tokens: Option<u64>,
    ) -> (Arc<Scheduler<I, T>>, bool) {
        let budget = if self.cost_pinned {
            None
        } else {
            model_max_batch_tokens.filter(|&b| b > 0)
        };
        let existing = self.schedulers.read().await.get(model_id).cloned();
        if let Some(sched) = existing {
            Self::adopt(model_id, &sched, budget).await;
            return (sched, false);
        }
        let mut map = self.schedulers.write().await;
        if let Some(sched) = map.get(model_id).cloned() {
            drop(map);
            Self::adopt(model_id, &sched, budget).await;
            return (sched, false);
        }
        let mut config = self.default_config;
        let source = if self.cost_pinned {
            "env"
        } else if let Some(budget) = budget {
            config.max_batch_cost = budget;
            "model"
        } else {
            "default"
        };
        let sched = Arc::new(
            Scheduler::builder()
                .config(config)
                .env_lookup(Arc::clone(&self.env))
                .build(),
        );
        let max_batch_cost = sched.config().await.max_batch_cost;
        info!(
            model = %model_id,
            budget = config.max_batch_cost,
            source,
            max_batch_cost,
            "rust-scheduler: scheduler created"
        );
        map.insert(model_id.to_owned(), Arc::clone(&sched));
        (sched, true)
    }

    async fn adopt(model_id: &str, sched: &Scheduler<I, T>, budget: Option<u64>) {
        let Some(budget) = budget else {
            return;
        };
        let previous = sched.cost_budget();
        if sched.adopt_cost_budget(budget).await {
            let max_batch_cost = sched.config().await.max_batch_cost;
            info!(
                model = %model_id,
                previous_budget = previous,
                budget,
                max_batch_cost,
                "rust-scheduler: batch cost budget changed; caps and adaptive range re-based"
            );
        }
    }

    /// List the model ids that currently have an instantiated
    /// scheduler. Useful for dashboards and for the shadow-trace
    /// replay which needs to enumerate live per-model state.
    pub async fn active_models(&self) -> Vec<String> {
        self.schedulers.read().await.keys().cloned().collect()
    }

    /// Number of active schedulers. Cheap — one read-lock acquire.
    pub async fn active_count(&self) -> usize {
        self.schedulers.read().await.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc_types::{ExtractBatchItem, PreparedAudioPcm16, WireValue};
    use crate::scheduler::engine::{LoraKey, Op};
    use crate::scheduler::item::SchedulerItem;

    #[derive(Debug, Clone, Copy)]
    struct StubItem;
    impl HasCost for StubItem {
        fn cost(&self) -> u64 {
            1
        }
        fn original_index(&self) -> usize {
            0
        }
    }

    #[tokio::test]
    async fn get_or_create_marks_first_call_as_created() {
        let reg: SchedulerRegistry<StubItem, ()> = SchedulerRegistry::new(BatchConfig::default());
        let (_, created1) = reg.get_or_create("foo", None).await;
        assert!(created1, "first call for a model must report created=true");
        let (_, created2) = reg.get_or_create("foo", None).await;
        assert!(
            !created2,
            "subsequent call for the same model must report created=false"
        );
    }

    #[tokio::test]
    async fn get_or_create_returns_same_arc_on_repeat_calls() {
        let reg: SchedulerRegistry<StubItem, ()> = SchedulerRegistry::new(BatchConfig::default());
        let (a, _) = reg.get_or_create("foo", None).await;
        let (b, _) = reg.get_or_create("foo", None).await;
        // Pointer equality — lazy creation must memoize the Arc so
        // adaptive-controller state persists across dispatcher calls.
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[tokio::test]
    async fn different_models_get_different_schedulers() {
        let reg: SchedulerRegistry<StubItem, ()> = SchedulerRegistry::new(BatchConfig::default());
        let (a, _) = reg.get_or_create("foo", None).await;
        let (b, _) = reg.get_or_create("bar", None).await;
        assert!(!Arc::ptr_eq(&a, &b));
    }

    #[tokio::test]
    async fn active_models_reports_materialised_schedulers() {
        let reg: SchedulerRegistry<StubItem, ()> = SchedulerRegistry::new(BatchConfig::default());
        assert!(reg.active_models().await.is_empty());
        let _ = reg.get_or_create("foo", None).await;
        let _ = reg.get_or_create("bar", None).await;
        let mut active = reg.active_models().await;
        active.sort();
        assert_eq!(active, vec!["bar".to_string(), "foo".to_string()]);
        assert_eq!(reg.active_count().await, 2);
    }

    // ---- Per-model batch cost budget ----

    const WHISPER: &str = "openai/whisper-large-v3-turbo";

    async fn cost_state<I, T>(sched: &Scheduler<I, T>) -> (u64, u64, u64)
    where
        I: HasCost + Send + Sync + 'static,
        T: Send + Sync + 'static,
    {
        (
            sched.cost_budget(),
            sched.config().await.max_batch_cost,
            sched.controller_snapshot().await.current_batch_cost,
        )
    }

    /// Advance the controller's starvation streak by `n` single-item
    /// batches: state that survives only while the same controller runs.
    async fn mark_controller<I, T>(sched: &Scheduler<I, T>, n: u32) -> u32
    where
        I: HasCost + Send + Sync + 'static,
        T: Send + Sync + 'static,
    {
        for _ in 0..n {
            let _ = sched.record_completion(1, 1).await;
        }
        sched.controller_snapshot().await.starvation_streak
    }

    #[tokio::test]
    async fn model_budget_sets_the_cost_cap_and_adaptive_start() {
        let reg: SchedulerRegistry<StubItem, ()> = SchedulerRegistry::new(BatchConfig::default());
        let (sched, created) = reg.get_or_create(WHISPER, Some(720_000)).await;
        assert!(created);
        assert_eq!(cost_state(&sched).await, (720_000, 720_000, 720_000));
        // Everything but the cost cap still comes from the default config.
        assert_eq!(
            sched.config().await,
            BatchConfig {
                max_batch_cost: 720_000,
                ..BatchConfig::default()
            }
        );
    }

    #[tokio::test]
    async fn missing_or_zero_model_budget_keeps_the_default_cap() {
        for reported in [None, Some(0)] {
            let reg: SchedulerRegistry<StubItem, ()> =
                SchedulerRegistry::new(BatchConfig::default());
            let (sched, _) = reg.get_or_create("m", reported).await;
            assert_eq!(cost_state(&sched).await, (16_384, 16_384, 16_384));
            assert_eq!(sched.config().await, BatchConfig::default());
        }
    }

    #[tokio::test]
    async fn a_16384_budget_runs_exactly_like_the_default() {
        // Most profiles declare 16384; their schedulers must not change.
        let reg: SchedulerRegistry<StubItem, ()> = SchedulerRegistry::new(BatchConfig::default());
        let (reported, _) = reg.get_or_create("reported", Some(16_384)).await;
        let (silent, _) = reg.get_or_create("silent", None).await;
        assert_eq!(reported.config().await, silent.config().await);
        assert_eq!(
            reported.controller_snapshot().await,
            silent.controller_snapshot().await
        );

        // Reporting 16384 to a scheduler created without a budget keeps
        // its controller: nothing re-calibrates.
        assert_eq!(mark_controller(&silent, 3).await, 3);
        let (again, _) = reg.get_or_create("silent", Some(16_384)).await;
        assert!(Arc::ptr_eq(&again, &silent));
        assert_eq!(again.controller_snapshot().await.starvation_streak, 3);
    }

    #[tokio::test]
    async fn cost_pinned_registry_ignores_model_budgets() {
        // `SIE_BATCHER_MAX_BATCH_COST` set: its cap wins for every model.
        let pinned = BatchConfig {
            max_batch_cost: 32_768,
            ..BatchConfig::default()
        };
        let reg: SchedulerRegistry<StubItem, ()> =
            SchedulerRegistry::new(pinned).with_cost_pinned(true);
        let (sched, _) = reg.get_or_create(WHISPER, Some(720_000)).await;
        assert_eq!(cost_state(&sched).await, (32_768, 32_768, 32_768));
        let (sched, _) = reg.get_or_create(WHISPER, Some(4_096)).await;
        assert_eq!(cost_state(&sched).await, (32_768, 32_768, 32_768));
    }

    #[tokio::test]
    async fn same_budget_again_keeps_the_scheduler_and_its_controller() {
        let reg: SchedulerRegistry<StubItem, ()> = SchedulerRegistry::new(BatchConfig::default());
        let (first, _) = reg.get_or_create(WHISPER, Some(720_000)).await;
        assert_eq!(mark_controller(&first, 3).await, 3);

        let (second, created) = reg.get_or_create(WHISPER, Some(720_000)).await;
        assert!(!created);
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(second.controller_snapshot().await.starvation_streak, 3);
        assert_eq!(cost_state(&second).await, (720_000, 720_000, 720_000));
    }

    #[tokio::test]
    async fn changed_budget_rebases_an_existing_scheduler() {
        // Created against a backend that reported nothing, then the
        // backend restarts and reports the model's budget.
        let reg: SchedulerRegistry<StubItem, ()> = SchedulerRegistry::new(BatchConfig::default());
        let (sched, _) = reg.get_or_create(WHISPER, None).await;
        sched
            .submit(Op::Extract, LoraKey::base(), StubItem, ())
            .await;
        assert_eq!(mark_controller(&sched, 3).await, 3);

        let (same, created) = reg.get_or_create(WHISPER, Some(720_000)).await;
        assert!(!created);
        assert!(Arc::ptr_eq(&sched, &same));
        assert_eq!(cost_state(&sched).await, (720_000, 720_000, 720_000));
        assert_eq!(
            sched.controller_snapshot().await.starvation_streak,
            0,
            "the controller is rebuilt for the new budget"
        );

        // `None` from a later answer never undoes an adopted budget.
        let _ = reg.get_or_create(WHISPER, None).await;
        assert_eq!(cost_state(&sched).await, (720_000, 720_000, 720_000));

        // A further change re-bases again.
        let _ = reg.get_or_create(WHISPER, Some(4_096)).await;
        assert_eq!(cost_state(&sched).await, (4_096, 4_096, 4_096));
    }

    /// An environment holding exactly `pairs`.
    fn env_table(pairs: &'static [(&'static str, &'static str)]) -> EnvLookup {
        Arc::new(move |var| {
            pairs
                .iter()
                .find(|(name, _)| *name == var)
                .map(|(_, value)| (*value).to_string())
        })
    }

    #[tokio::test]
    async fn env_cost_cap_wins_over_reported_budgets() {
        // Set: every model keeps the operator's cap and the range it derives.
        let reg: SchedulerRegistry<StubItem, ()> =
            SchedulerRegistry::from_lookup(env_table(&[("SIE_BATCHER_MAX_BATCH_COST", "32768")]));
        let (sched, _) = reg.get_or_create(WHISPER, Some(720_000)).await;
        assert_eq!(cost_state(&sched).await, (32_768, 32_768, 32_768));
        let (sched, _) = reg.get_or_create("text-model", None).await;
        assert_eq!(cost_state(&sched).await, (32_768, 32_768, 32_768));

        // Unset or unparseable: the reported budget applies, else 16384.
        for pairs in [&[][..], &[("SIE_BATCHER_MAX_BATCH_COST", "16k")][..]] {
            let reg: SchedulerRegistry<StubItem, ()> =
                SchedulerRegistry::from_lookup(env_table(pairs));
            let (sched, _) = reg.get_or_create(WHISPER, Some(720_000)).await;
            assert_eq!(
                cost_state(&sched).await,
                (720_000, 720_000, 720_000),
                "{pairs:?}"
            );
            let (sched, _) = reg.get_or_create("text-model", None).await;
            assert_eq!(
                cost_state(&sched).await,
                (16_384, 16_384, 16_384),
                "{pairs:?}"
            );
        }
    }

    #[tokio::test]
    async fn adaptive_cost_ceiling_bounds_batches_from_the_first_one() {
        // `SIE_ADAPTIVE_BATCH_MAX_COST` holds before any controller step:
        // after creation and after a re-base, not only after a completion.
        let reg: SchedulerRegistry<SchedulerItem, String> =
            SchedulerRegistry::from_lookup(env_table(&[("SIE_ADAPTIVE_BATCH_MAX_COST", "65536")]));
        let base = LoraKey::base;
        let clips = || {
            (0..12)
                .map(|i| (audio(25_000), format!("clip-{i}")))
                .collect::<Vec<_>>()
        };

        let (sched, _) = reg.get_or_create(WHISPER, Some(720_000)).await;
        assert_eq!(cost_state(&sched).await, (720_000, 65_536, 65_536));
        sched.submit_many(Op::Extract, base(), clips()).await;
        let first = sched
            .try_drain_same(Op::Extract, base())
            .await
            .expect("clips pending");
        assert_eq!((first.size(), first.total_cost), (2, 50_000));

        // Re-based from the default budget onto the model's.
        let (other, _) = reg.get_or_create("other-speech-model", None).await;
        assert_eq!(cost_state(&other).await, (16_384, 16_384, 16_384));
        other.submit_many(Op::Extract, base(), clips()).await;
        let _ = reg.get_or_create("other-speech-model", Some(720_000)).await;
        assert_eq!(cost_state(&other).await, (720_000, 65_536, 65_536));
        let batch = other
            .try_drain_same(Op::Extract, base())
            .await
            .expect("clips pending");
        assert_eq!((batch.size(), batch.total_cost), (2, 50_000));
    }

    // ---- The clip-starvation case behind per-model budgets ----

    /// A sidecar-prepared 16 kHz audio extract item lasting `ms`.
    fn audio(ms: u64) -> SchedulerItem {
        SchedulerItem::Extract(ExtractBatchItem {
            work_item_id: "r.0".into(),
            request_id: "r".into(),
            item_index: 0,
            total_items: 1,
            timestamp: 0.0,
            item: WireValue::Nil,
            labels: None,
            output_schema: None,
            instruction: None,
            options: None,
            profile_id: None,
            bundle_config_hash: None,
            payload_fetch_ms: 0.0,
            prepared_audio: Some(PreparedAudioPcm16 {
                pcm_s16le: vec![],
                sample_rate: 16_000,
                sample_count: ms * 16,
                duration_ms: ms,
                source_sample_rate: 16_000,
                source_sample_count: ms * 16,
                source_channels: 1,
                container: "wav".into(),
            }),
        })
    }

    /// Long recordings and saturating clips on one speech model: the
    /// lanes alternate, so the clips one clip turn takes are the clip
    /// throughput between two recordings. A turn takes the clips pending
    /// when it began up to the cost cap. With the shared 16384 ms default
    /// that was 1-2 clips; with the model's own 720000 ms budget it is
    /// every pending clip, in batches of at most the request count cap.
    #[tokio::test]
    async fn model_budget_lets_one_clip_turn_take_the_pending_clips() {
        // 16 clips of 6-15 s: 152 s of audio, far over 16384 ms, and more
        // clips than the request count cap (12) takes in one batch.
        let clip_seconds = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 6, 7, 8, 9, 10, 11];
        for (budget, clip_turn) in [
            (None, vec![vec![6, 6]]),
            (
                Some(720_000),
                vec![
                    vec![6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11],
                    vec![12, 13, 14, 15],
                ],
            ),
        ] {
            let reg: SchedulerRegistry<SchedulerItem, String> =
                SchedulerRegistry::new(BatchConfig::default());
            let (sched, _) = reg.get_or_create(WHISPER, budget).await;
            let base = LoraKey::base;

            sched
                .submit(Op::Extract, base(), audio(240_000), "long-1".into())
                .await;
            // Keep arrivals ordered: the recordings bracket the clips.
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
            let clips = clip_seconds
                .iter()
                .map(|&s| (audio(s * 1_000), format!("clip-{s}s")))
                .collect();
            sched.submit_many(Op::Extract, base(), clips).await;
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
            sched
                .submit(Op::Extract, base(), audio(200_000), "long-2".into())
                .await;

            let mut served = Vec::new();
            while let Some(batch) = sched.try_drain_same(Op::Extract, base()).await {
                served.push(batch.metadata);
            }

            let seconds = |names: &[String]| -> Vec<u64> {
                names
                    .iter()
                    .map(|name| {
                        name.trim_start_matches("clip-")
                            .trim_end_matches('s')
                            .parse()
                            .expect("a clip")
                    })
                    .collect()
            };
            let turn_batches = clip_turn.len();
            assert_eq!(served[0], ["long-1"], "budget {budget:?}");
            let turn: Vec<Vec<u64>> = served[1..=turn_batches]
                .iter()
                .map(|batch| seconds(batch))
                .collect();
            assert_eq!(turn, clip_turn, "budget {budget:?}");
            assert_eq!(served[turn_batches + 1], ["long-2"], "budget {budget:?}");
            let clips_served: usize = served.iter().map(Vec::len).sum::<usize>() - 2;
            assert_eq!(clips_served, clip_seconds.len(), "budget {budget:?}");
            if budget.is_some() {
                assert_eq!(served.len(), 4, "one clip turn serves every clip");
            } else {
                assert!(served.len() > 4, "clips are still pending after one turn");
            }
        }
    }
}
