//! [`BackendRouter`] — picks the right [`InferenceBackend`] for a
//! given `model_id` and forwards the call.
//!
//! # Routing rules
//!
//! * Backends are tried in **construction order**. The first backend
//!   whose `supports(model_id)` returns `true` wins.
//! * Routing is cached per `model_id` after the first successful
//!   [`InferenceBackend::ensure_model_ready`] so we don't re-scan on
//!   every batch. A backend that later stops claiming a model is not
//!   handled — restart the worker.
//! * If `ensure_model_ready` returns [`BackendError::UnsupportedModel`],
//!   the router falls through to the next backend. Any **other** error
//!   short-circuits: we do not silently swap backends on transient
//!   failures (numeric divergence between two embedders = correctness
//!   bug, not feature).
//!
//! # Configuration
//!
//! Callers construct the router explicitly via
//! [`BackendRouter::from_backends`]. Priority comes from the caller
//! (see `lib.rs::run`); this module stays focused on dispatch.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;
use tracing::{debug, warn};

use crate::backend::{BackendError, InferenceBackend, SharedBackend};
use crate::ipc_types::{
    BatchOutcome, EnsureModelReadyResponse, ProcessEncodeBatchRequest, ProcessExtractBatchRequest,
    ProcessScoreBatchRequest, RunBatchRequest,
};

/// Picks a backend per `model_id` and forwards calls to it.
pub struct BackendRouter {
    /// Ordered list — first match wins.
    backends: Vec<SharedBackend>,
    /// `model_id -> index into backends`. Populated on first
    /// successful readiness check; avoids re-scanning on every batch.
    cache: RwLock<HashMap<String, usize>>,
}

impl BackendRouter {
    /// Construct from a non-empty, priority-ordered backend list.
    ///
    /// # Panics
    ///
    /// If `backends` is empty — a router with no backends is always
    /// wrong, and silently accepting that just moves the error to
    /// runtime where it'd NAK every message.
    pub fn from_backends(backends: Vec<SharedBackend>) -> Arc<Self> {
        assert!(
            !backends.is_empty(),
            "BackendRouter requires at least one backend"
        );
        Arc::new(Self {
            backends,
            cache: RwLock::new(HashMap::new()),
        })
    }

    /// Convenience for the default (Python-only) deployment.
    pub fn with_python_only(python: SharedBackend) -> Arc<Self> {
        Self::from_backends(vec![python])
    }

    async fn cached_index(&self, model_id: &str) -> Option<usize> {
        self.cache.read().await.get(model_id).copied()
    }

    async fn cache_index(&self, model_id: &str, idx: usize) {
        self.cache.write().await.insert(model_id.to_string(), idx);
    }

    async fn invalidate_index(&self, model_id: &str) {
        self.cache.write().await.remove(model_id);
    }

    /// First backend that claims `model_id` via `supports()`.
    fn first_supporting(&self, model_id: &str) -> Option<usize> {
        self.backends.iter().position(|b| b.supports(model_id))
    }

    /// Resolve a backend for `model_id`, consulting (and populating)
    /// the cache. Returns `UnsupportedModel` if no backend claims it.
    ///
    /// Returns both the index and the backend so callers that need to
    /// fall through on `UnsupportedModel` at the `process_*_batch`
    /// level can start the search from the next slot.
    async fn backend_for(&self, model_id: &str) -> Result<(usize, &SharedBackend), BackendError> {
        if let Some(idx) = self.cached_index(model_id).await {
            return Ok((idx, &self.backends[idx]));
        }
        let idx = self
            .first_supporting(model_id)
            .ok_or_else(|| BackendError::UnsupportedModel(model_id.to_string()))?;
        Ok((idx, &self.backends[idx]))
    }

    /// Generic dispatch for a per-op `process_*_batch` call with
    /// fall-through on `UnsupportedModel`. Mirrors the
    /// `ensure_model_ready` contract: a backend that claims a model via
    /// `supports()` but rejects a specific op (e.g. a cross-encoder
    /// asked to `encode`) is treated as a routing miss, not a hard
    /// failure — the router tries the next backend. Any other error
    /// short-circuits.
    async fn dispatch<F, Fut>(
        &self,
        model_id: &str,
        op_name: &'static str,
        batch_size: usize,
        call: F,
    ) -> Result<BatchOutcome, BackendError>
    where
        F: Fn(SharedBackend) -> Fut,
        Fut: std::future::Future<Output = Result<BatchOutcome, BackendError>>,
    {
        let (start_idx, first) = self.backend_for(model_id).await?;
        let first_name = first.name();
        match self
            .call_instrumented(Arc::clone(first), model_id, op_name, batch_size, &call)
            .await
        {
            Ok(resp) => return Ok(resp),
            Err(BackendError::UnsupportedModel(_)) => {
                // The backend claimed the model via `supports()` but
                // rejected this specific op (classic shape: a native
                // encoder asked to `score` — delegate to Python IPC).
                // Do NOT invalidate the cached binding: the cache is
                // per-model, so wiping it would also reroute future
                // `encode` calls through Python and defeat the native
                // fast path.
                debug!(
                    model = %model_id,
                    backend = first_name,
                    op = op_name,
                    "router: backend doesn't serve this op, falling through"
                );
            }
            Err(e) => return Err(e),
        }

        for (_idx, backend) in self.backends.iter().enumerate().skip(start_idx + 1) {
            if !backend.supports(model_id) {
                continue;
            }
            match self
                .call_instrumented(Arc::clone(backend), model_id, op_name, batch_size, &call)
                .await
            {
                Ok(resp) => {
                    // NOTE: Do NOT promote this backend in the cache. The
                    // model cache is per-model, not per-(model, op). A
                    // common deployment has a native encoder serve
                    // `encode` for a given model_id and Python IPC serve
                    // `score` / `extract` for the same model_id — if we
                    // cached the Python backend after the first score
                    // call, subsequent `encode` calls would also route
                    // to Python, silently losing the native fast path.
                    // Re-resolving from `first_supporting` on every miss
                    // keeps ops routing correct at the cost of a cheap
                    // O(backends) scan per op-level miss.
                    return Ok(resp);
                }
                Err(BackendError::UnsupportedModel(_)) => continue,
                Err(e) => return Err(e),
            }
        }

        Err(BackendError::UnsupportedModel(model_id.to_string()))
    }

    async fn ensure_ready_instrumented(
        &self,
        backend: &SharedBackend,
        model_id: &str,
    ) -> Result<EnsureModelReadyResponse, BackendError> {
        backend.ensure_model_ready(model_id).await
    }

    /// Invoke one backend and record the per-backend observability
    /// metrics around it. Keeps `dispatch` readable while ensuring
    /// every call site gets uniform instrumentation.
    async fn call_instrumented<F, Fut>(
        &self,
        backend: SharedBackend,
        _model_id: &str,
        _op_name: &'static str,
        _batch_size: usize,
        call: &F,
    ) -> Result<BatchOutcome, BackendError>
    where
        F: Fn(SharedBackend) -> Fut,
        Fut: std::future::Future<Output = Result<BatchOutcome, BackendError>>,
    {
        call(backend).await
    }

    /// Total number of registered backends. Always `>= 1` —
    /// [`BackendRouter::from_backends`] panics on empty input, so
    /// the `is_empty` sibling that clippy would like us to expose
    /// would always return `false`.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.backends.len()
    }

    /// Drain every registered backend concurrently; one slow drain
    /// can't block the others.
    ///
    /// Dedupes by `Arc` pointer identity as a defensive measure in
    /// case the same backend was registered twice (e.g. an ops misconfig).
    pub async fn drain_all(&self, deadline_ms: u64) {
        use std::collections::HashSet;
        // Arc pointer identity as `usize` so the set is `Send`.
        let mut seen: HashSet<usize> = HashSet::new();
        let mut unique: Vec<Arc<dyn InferenceBackend>> = Vec::with_capacity(self.backends.len());
        for b in &self.backends {
            let id = Arc::as_ptr(b) as *const () as usize;
            if seen.insert(id) {
                unique.push(Arc::clone(b));
            }
        }
        // Drop `seen` before the await so the future stays Send.
        drop(seen);
        let futs: Vec<_> = unique
            .into_iter()
            .map(|b| async move { b.drain(deadline_ms).await })
            .collect();
        futures_util::future::join_all(futs).await;
    }

    /// Backend names, in priority order. For logging at startup.
    pub fn names(&self) -> Vec<&'static str> {
        self.backends.iter().map(|b| b.name()).collect()
    }
}

/// The router itself satisfies [`InferenceBackend`], so the dispatcher
/// can hold an `Arc<BackendRouter>` or `Arc<dyn InferenceBackend>`
/// interchangeably.
#[async_trait]
impl InferenceBackend for BackendRouter {
    fn name(&self) -> &'static str {
        "router"
    }

    fn supports(&self, model_id: &str) -> bool {
        self.backends.iter().any(|b| b.supports(model_id))
    }

    async fn ensure_model_ready(
        &self,
        model_id: &str,
    ) -> Result<EnsureModelReadyResponse, BackendError> {
        let mut start_idx = 0;
        if let Some(idx) = self.cached_index(model_id).await {
            match self
                .ensure_ready_instrumented(&self.backends[idx], model_id)
                .await
            {
                Ok(resp) => return Ok(resp),
                Err(BackendError::UnsupportedModel(_)) => {
                    self.invalidate_index(model_id).await;
                    start_idx = idx + 1;
                }
                Err(e) => return Err(e),
            }
        }

        for (idx, backend) in self.backends.iter().enumerate().skip(start_idx) {
            if !backend.supports(model_id) {
                continue;
            }
            match self.ensure_ready_instrumented(backend, model_id).await {
                Ok(resp) => {
                    debug!(
                        model = %model_id,
                        backend = backend.name(),
                        state = ?resp.state,
                        "router: bound model to backend"
                    );
                    self.cache_index(model_id, idx).await;
                    return Ok(resp);
                }
                Err(BackendError::UnsupportedModel(_)) => continue,
                Err(e) => {
                    // Hard error on the first claimant — do NOT silently
                    // fall through to the next backend. Surface so the
                    // dispatcher NAKs and ops can diagnose.
                    warn!(
                        model = %model_id,
                        backend = backend.name(),
                        error = %e,
                        "router: backend readiness failed, not falling through"
                    );
                    return Err(e);
                }
            }
        }

        Err(BackendError::UnsupportedModel(model_id.to_string()))
    }

    async fn process_encode_batch(
        &self,
        req: ProcessEncodeBatchRequest,
    ) -> Result<BatchOutcome, BackendError> {
        let model_id = req.model_id.clone();
        let batch_size = req.items.len();
        // Cheap to clone in the common (single-backend) case; the clone
        // only matters if we fall through, which is the rare path.
        let result = self
            .dispatch(&model_id, "encode", batch_size, move |backend| {
                let req = req.clone();
                async move { backend.process_encode_batch(req).await }
            })
            .await;
        result
    }

    async fn process_score_batch(
        &self,
        req: ProcessScoreBatchRequest,
    ) -> Result<BatchOutcome, BackendError> {
        let model_id = req.model_id.clone();
        let batch_size = req.items.len();
        let result = self
            .dispatch(&model_id, "score", batch_size, move |backend| {
                let req = req.clone();
                async move { backend.process_score_batch(req).await }
            })
            .await;
        result
    }

    async fn process_extract_batch(
        &self,
        req: ProcessExtractBatchRequest,
    ) -> Result<BatchOutcome, BackendError> {
        let model_id = req.model_id.clone();
        let batch_size = req.items.len();
        let result = self
            .dispatch(&model_id, "extract", batch_size, move |backend| {
                let req = req.clone();
                async move { backend.process_extract_batch(req).await }
            })
            .await;
        result
    }

    async fn run_batch(&self, req: RunBatchRequest) -> Result<BatchOutcome, BackendError> {
        // `run_batch` uses the same fall-through semantics as the
        // per-op dispatchers: the first backend whose `supports()`
        // claims `model_id` handles it; `UnsupportedModel` trips the
        // next backend in order. A native backend that only implements
        // the per-op `process_*_batch` calls returns `UnsupportedModel`
        // from `run_batch` (the default impl), so this path lets the
        // Python IPC catch-all (tail of the router chain) pick up the
        // batch without forcing every native backend to implement
        // `run_batch`.
        let model_id = req.model_id.clone();
        let batch_size = req.items.len();
        let result = self
            .dispatch(&model_id, "run_batch", batch_size, move |backend| {
                let req = req.clone();
                async move { backend.run_batch(req).await }
            })
            .await;
        result
    }

    async fn drain(&self, deadline_ms: u64) {
        self.drain_all(deadline_ms).await;
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc_types::ReadinessState;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Test double — claims a configurable set of models and counts calls.
    ///
    /// `supports_all` makes it claim every model (mimicking Python).
    struct MockBackend {
        name: &'static str,
        supported: Vec<String>,
        supports_all: bool,
        ready_err: Option<BackendError>,
        process_err: Option<BackendError>,
        score_err: Option<BackendError>,
        extract_err: Option<BackendError>,
        ready_calls: AtomicUsize,
        process_calls: AtomicUsize,
        score_calls: AtomicUsize,
        extract_calls: AtomicUsize,
        drain_calls: AtomicUsize,
    }

    impl MockBackend {
        fn scoped(name: &'static str, supported: &[&str]) -> Self {
            Self {
                name,
                supported: supported.iter().map(|s| s.to_string()).collect(),
                supports_all: false,
                ready_err: None,
                process_err: None,
                score_err: None,
                extract_err: None,
                ready_calls: AtomicUsize::new(0),
                process_calls: AtomicUsize::new(0),
                score_calls: AtomicUsize::new(0),
                extract_calls: AtomicUsize::new(0),
                drain_calls: AtomicUsize::new(0),
            }
        }
        fn all(name: &'static str) -> Self {
            Self {
                name,
                supported: vec![],
                supports_all: true,
                ready_err: None,
                process_err: None,
                score_err: None,
                extract_err: None,
                ready_calls: AtomicUsize::new(0),
                process_calls: AtomicUsize::new(0),
                score_calls: AtomicUsize::new(0),
                extract_calls: AtomicUsize::new(0),
                drain_calls: AtomicUsize::new(0),
            }
        }
        fn with_ready_error(mut self, err: BackendError) -> Self {
            self.ready_err = Some(err);
            self
        }
        fn with_process_error(mut self, err: BackendError) -> Self {
            self.process_err = Some(err);
            self
        }
        fn with_score_error(mut self, err: BackendError) -> Self {
            self.score_err = Some(err);
            self
        }
        fn with_extract_error(mut self, err: BackendError) -> Self {
            self.extract_err = Some(err);
            self
        }
    }

    fn clone_err(err: &BackendError) -> BackendError {
        match err {
            BackendError::Transient(m) => BackendError::Transient(m.clone()),
            BackendError::Inference(m) => BackendError::Inference(m.clone()),
            BackendError::UnsupportedModel(m) => BackendError::UnsupportedModel(m.clone()),
            BackendError::Draining => BackendError::Draining,
        }
    }

    #[async_trait]
    impl InferenceBackend for MockBackend {
        fn name(&self) -> &'static str {
            self.name
        }
        fn supports(&self, model_id: &str) -> bool {
            self.supports_all || self.supported.iter().any(|m| m == model_id)
        }
        async fn ensure_model_ready(
            &self,
            _model_id: &str,
        ) -> Result<EnsureModelReadyResponse, BackendError> {
            self.ready_calls.fetch_add(1, Ordering::SeqCst);
            if let Some(err) = &self.ready_err {
                return Err(clone_err(err));
            }
            Ok(EnsureModelReadyResponse {
                state: ReadinessState::Ready,
                batch_budget: Some(32),
                descriptor: None,
            })
        }
        async fn process_encode_batch(
            &self,
            _req: ProcessEncodeBatchRequest,
        ) -> Result<BatchOutcome, BackendError> {
            self.process_calls.fetch_add(1, Ordering::SeqCst);
            if let Some(err) = &self.process_err {
                return Err(clone_err(err));
            }
            Ok(BatchOutcome {
                outcomes: vec![],
                batched_f16_multivectors: vec![],
            })
        }
        async fn process_score_batch(
            &self,
            _req: ProcessScoreBatchRequest,
        ) -> Result<BatchOutcome, BackendError> {
            self.score_calls.fetch_add(1, Ordering::SeqCst);
            if let Some(err) = self.score_err.as_ref().or(self.process_err.as_ref()) {
                return Err(clone_err(err));
            }
            Ok(BatchOutcome {
                outcomes: vec![],
                batched_f16_multivectors: vec![],
            })
        }
        async fn process_extract_batch(
            &self,
            _req: ProcessExtractBatchRequest,
        ) -> Result<BatchOutcome, BackendError> {
            self.extract_calls.fetch_add(1, Ordering::SeqCst);
            if let Some(err) = self.extract_err.as_ref().or(self.process_err.as_ref()) {
                return Err(clone_err(err));
            }
            Ok(BatchOutcome {
                outcomes: vec![],
                batched_f16_multivectors: vec![],
            })
        }
        async fn drain(&self, _deadline_ms: u64) {
            self.drain_calls.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn encode_req(model: &str) -> ProcessEncodeBatchRequest {
        ProcessEncodeBatchRequest {
            model_id: model.into(),
            items: vec![],
            accepts_batched_f16_multivectors: false,
        }
    }

    #[tokio::test]
    async fn single_backend_owns_everything() {
        let b = Arc::new(MockBackend::all("python")) as SharedBackend;
        let router = BackendRouter::with_python_only(b);
        let resp = router.ensure_model_ready("any-model").await.unwrap();
        assert!(matches!(resp.state, ReadinessState::Ready));
        assert_eq!(router.len(), 1);
    }

    #[tokio::test]
    async fn native_backend_wins_over_python_when_it_supports_model() {
        let native = Arc::new(MockBackend::scoped("native", &["bge-base"]));
        let python = Arc::new(MockBackend::all("python"));
        let router = BackendRouter::from_backends(vec![
            Arc::clone(&native) as SharedBackend,
            Arc::clone(&python) as SharedBackend,
        ]);

        let resp = router.ensure_model_ready("bge-base").await.unwrap();
        assert!(matches!(resp.state, ReadinessState::Ready));
        assert_eq!(native.ready_calls.load(Ordering::SeqCst), 1);
        assert_eq!(python.ready_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn falls_through_to_python_when_native_does_not_support() {
        let native = Arc::new(MockBackend::scoped("native", &["bge-base"]));
        let python = Arc::new(MockBackend::all("python"));
        let router = BackendRouter::from_backends(vec![
            Arc::clone(&native) as SharedBackend,
            Arc::clone(&python) as SharedBackend,
        ]);

        let resp = router.ensure_model_ready("gpt-neo").await.unwrap();
        assert!(matches!(resp.state, ReadinessState::Ready));
        assert_eq!(native.ready_calls.load(Ordering::SeqCst), 0);
        assert_eq!(python.ready_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn native_hard_error_does_not_fall_through_to_python() {
        // Correctness invariant: if a native backend claims a model
        // and then its readiness call fails with a non-Unsupported
        // error, we MUST surface that error. Silently retrying via
        // Python would let numeric drift leak into production.
        let native = Arc::new(
            MockBackend::scoped("native", &["bge-base"])
                .with_ready_error(BackendError::Transient("cuda oom".into())),
        );
        let python = Arc::new(MockBackend::all("python"));
        let router = BackendRouter::from_backends(vec![
            Arc::clone(&native) as SharedBackend,
            Arc::clone(&python) as SharedBackend,
        ]);

        let err = router.ensure_model_ready("bge-base").await.unwrap_err();
        assert!(matches!(err, BackendError::Transient(_)));
        assert_eq!(python.ready_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn native_unsupported_falls_through_to_next_backend() {
        // If a backend claims via supports() but then returns
        // UnsupportedModel at readiness time (e.g. the weights hash
        // didn't match), the router should try the next backend.
        let native = Arc::new(
            MockBackend::scoped("native", &["bge-base"])
                .with_ready_error(BackendError::UnsupportedModel("bge-base".into())),
        );
        let python = Arc::new(MockBackend::all("python"));
        let router = BackendRouter::from_backends(vec![
            Arc::clone(&native) as SharedBackend,
            Arc::clone(&python) as SharedBackend,
        ]);

        let resp = router.ensure_model_ready("bge-base").await.unwrap();
        assert!(matches!(resp.state, ReadinessState::Ready));
        assert_eq!(native.ready_calls.load(Ordering::SeqCst), 1);
        assert_eq!(python.ready_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn cached_backend_unsupported_falls_through_to_next_backend() {
        let native = Arc::new(
            MockBackend::scoped("native", &["bge-base"])
                .with_ready_error(BackendError::UnsupportedModel("bge-base".into())),
        );
        let python = Arc::new(MockBackend::all("python"));
        let router = BackendRouter::from_backends(vec![
            Arc::clone(&native) as SharedBackend,
            Arc::clone(&python) as SharedBackend,
        ]);
        router.cache_index("bge-base", 0).await;

        let resp = router.ensure_model_ready("bge-base").await.unwrap();
        assert!(matches!(resp.state, ReadinessState::Ready));
        assert_eq!(native.ready_calls.load(Ordering::SeqCst), 1);
        assert_eq!(python.ready_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn unsupported_when_no_backend_claims() {
        let native = Arc::new(MockBackend::scoped("native", &["bge-base"])) as SharedBackend;
        let router = BackendRouter::from_backends(vec![native]);
        let err = router.ensure_model_ready("unknown").await.unwrap_err();
        assert!(matches!(err, BackendError::UnsupportedModel(_)));
    }

    #[tokio::test]
    async fn routes_process_encode_to_cached_backend() {
        let native = Arc::new(MockBackend::scoped("native", &["bge-base"]));
        let python = Arc::new(MockBackend::all("python"));
        let router = BackendRouter::from_backends(vec![
            Arc::clone(&native) as SharedBackend,
            Arc::clone(&python) as SharedBackend,
        ]);

        router.ensure_model_ready("bge-base").await.unwrap();
        let _ = router
            .process_encode_batch(encode_req("bge-base"))
            .await
            .unwrap();

        assert_eq!(native.process_calls.load(Ordering::SeqCst), 1);
        assert_eq!(python.process_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn process_without_prior_ready_still_routes_by_supports() {
        // Callers may hit process_* before ensure_model_ready (on
        // restart, or in tests). The router should still resolve via
        // supports() — just without caching the binding.
        let native = Arc::new(MockBackend::scoped("native", &["bge-base"]));
        let router = BackendRouter::from_backends(vec![Arc::clone(&native) as SharedBackend]);
        let _ = router
            .process_encode_batch(encode_req("bge-base"))
            .await
            .unwrap();
        assert_eq!(native.process_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn process_unknown_model_returns_unsupported() {
        let native = Arc::new(MockBackend::scoped("native", &["bge-base"])) as SharedBackend;
        let router = BackendRouter::from_backends(vec![native]);
        let err = router
            .process_encode_batch(encode_req("unknown"))
            .await
            .unwrap_err();
        assert!(matches!(err, BackendError::UnsupportedModel(_)));
    }

    #[tokio::test]
    async fn drain_all_calls_every_backend() {
        let a = Arc::new(MockBackend::all("a"));
        let b = Arc::new(MockBackend::all("b"));
        let router = BackendRouter::from_backends(vec![
            Arc::clone(&a) as SharedBackend,
            Arc::clone(&b) as SharedBackend,
        ]);
        router.drain_all(1000).await;
        assert_eq!(a.drain_calls.load(Ordering::SeqCst), 1);
        assert_eq!(b.drain_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn supports_is_union_over_backends() {
        let native = Arc::new(MockBackend::scoped("native", &["bge-base"])) as SharedBackend;
        let python = Arc::new(MockBackend::all("python")) as SharedBackend;
        let router = BackendRouter::from_backends(vec![Arc::clone(&native), Arc::clone(&python)]);
        // Both the specific and arbitrary models are supported, thanks
        // to Python claiming everything.
        assert!(router.supports("bge-base"));
        assert!(router.supports("anything-at-all"));
    }

    #[test]
    #[should_panic(expected = "at least one backend")]
    fn empty_router_panics() {
        let _ = BackendRouter::from_backends(Vec::new());
    }

    #[tokio::test]
    async fn process_falls_through_on_unsupported_op() {
        // Concrete case: a cross-encoder backend claims reranker
        // `model_id` via supports() (because it serves score for it),
        // but when asked to encode returns UnsupportedModel. The
        // router must try the next backend instead of surfacing the
        // error, which would cause a spurious NAK.
        let cross_encoder = Arc::new(
            MockBackend::scoped("xencoder", &["BAAI/bge-reranker-v2-m3"]).with_process_error(
                BackendError::UnsupportedModel("BAAI/bge-reranker-v2-m3".into()),
            ),
        );
        let python = Arc::new(MockBackend::all("python"));
        let router = BackendRouter::from_backends(vec![
            Arc::clone(&cross_encoder) as SharedBackend,
            Arc::clone(&python) as SharedBackend,
        ]);

        let _ = router
            .process_encode_batch(encode_req("BAAI/bge-reranker-v2-m3"))
            .await
            .unwrap();

        assert_eq!(cross_encoder.process_calls.load(Ordering::SeqCst), 1);
        assert_eq!(python.process_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn process_hard_error_does_not_fall_through() {
        // Inference / Transient errors must short-circuit — silently
        // retrying on the next backend would hide numeric drift.
        let cross_encoder = Arc::new(
            MockBackend::scoped("xencoder", &["reranker"])
                .with_process_error(BackendError::Transient("cuda oom".into())),
        );
        let python = Arc::new(MockBackend::all("python"));
        let router = BackendRouter::from_backends(vec![
            Arc::clone(&cross_encoder) as SharedBackend,
            Arc::clone(&python) as SharedBackend,
        ]);

        let err = router
            .process_encode_batch(encode_req("reranker"))
            .await
            .unwrap_err();
        assert!(matches!(err, BackendError::Transient(_)));
        assert_eq!(cross_encoder.process_calls.load(Ordering::SeqCst), 1);
        assert_eq!(python.process_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn encode_stays_on_native_after_score_falls_through_to_python() {
        // Deployment shape: the native encoder serves `encode` for a
        // model_id; the same model_id's `score` / `extract` must fall
        // through to Python IPC. Crucially, after the first `score`
        // call falls through, a subsequent `encode` call MUST still
        // land on the native backend — otherwise we silently lose the
        // native fast path for the remainder of the pod's lifetime.
        use crate::ipc_types::{ProcessExtractBatchRequest, ProcessScoreBatchRequest};

        let model = "BAAI/bge-base-en-v1.5";
        let native = Arc::new(
            MockBackend::scoped("native", &[model])
                .with_score_error(BackendError::UnsupportedModel(model.into()))
                .with_extract_error(BackendError::UnsupportedModel(model.into())),
        );
        let python = Arc::new(MockBackend::all("python"));
        let router = BackendRouter::from_backends(vec![
            Arc::clone(&native) as SharedBackend,
            Arc::clone(&python) as SharedBackend,
        ]);

        // First encode goes to native.
        router
            .process_encode_batch(encode_req(model))
            .await
            .unwrap();
        assert_eq!(native.process_calls.load(Ordering::SeqCst), 1);
        assert_eq!(python.process_calls.load(Ordering::SeqCst), 0);

        // Score falls through to python.
        let score_req = ProcessScoreBatchRequest {
            model_id: model.into(),
            items: vec![],
        };
        router.process_score_batch(score_req).await.unwrap();
        assert_eq!(native.score_calls.load(Ordering::SeqCst), 1);
        assert_eq!(python.score_calls.load(Ordering::SeqCst), 1);

        // Extract also falls through to python.
        let extract_req = ProcessExtractBatchRequest {
            model_id: model.into(),
            items: vec![],
        };
        router.process_extract_batch(extract_req).await.unwrap();
        assert_eq!(native.extract_calls.load(Ordering::SeqCst), 1);
        assert_eq!(python.extract_calls.load(Ordering::SeqCst), 1);

        // Subsequent encode must STILL go to the native backend — the
        // fall-through for score / extract must NOT have promoted
        // python in the per-model cache.
        router
            .process_encode_batch(encode_req(model))
            .await
            .unwrap();
        assert_eq!(
            native.process_calls.load(Ordering::SeqCst),
            2,
            "encode regressed to python after score fall-through"
        );
        assert_eq!(
            python.process_calls.load(Ordering::SeqCst),
            0,
            "encode should not have hit python"
        );
    }

    #[tokio::test]
    async fn process_no_other_claimant_surfaces_unsupported() {
        // Only the cross-encoder claims the id and only it would be
        // asked; if it rejects and no other backend supports the id,
        // we return UnsupportedModel to the dispatcher.
        let cross_encoder = Arc::new(
            MockBackend::scoped("xencoder", &["reranker"])
                .with_process_error(BackendError::UnsupportedModel("reranker".into())),
        );
        let router =
            BackendRouter::from_backends(vec![Arc::clone(&cross_encoder) as SharedBackend]);

        let err = router
            .process_encode_batch(encode_req("reranker"))
            .await
            .unwrap_err();
        assert!(matches!(err, BackendError::UnsupportedModel(_)));
        assert_eq!(cross_encoder.process_calls.load(Ordering::SeqCst), 1);
    }

    /// Repeated readiness checks re-handshake with the selected backend.
    /// The router caches model-to-backend binding, not the readiness
    /// response itself, so descriptor/batch-budget metadata can change
    /// safely after live config reconciliation.
    #[tokio::test]
    async fn repeated_ready_checks_call_backend() {
        let python = Arc::new(MockBackend::all("python"));
        let router = BackendRouter::from_backends(vec![Arc::clone(&python) as SharedBackend]);

        router.ensure_model_ready("model-a").await.unwrap();
        router.ensure_model_ready("model-a").await.unwrap();
        assert_eq!(python.ready_calls.load(Ordering::SeqCst), 2);
    }
}
