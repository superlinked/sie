//! [`BatchFormer`] — async pending queue + flush loop.
//!
//! Ported from `sie_server/core/batcher.py::BatchFormer`. The port preserves
//! its batching/packing contract, with the intentional Rust differences
//! called out below:
//!
//! 1. **Flush triggers**: cost cap, count cap, `max_batch_wait_ms`
//!    timeout since the *first* item, or the coalesce window
//!    (`effective_coalesce_ms`) elapsing since the *last* submit.
//! 2. **Oversize item**: a single item whose cost already exceeds
//!    `max_batch_cost` still flushes (alone) so it can't starve
//!    forever.
//! 3. **Bounded cost-sorted packing**: before each extract, the oldest
//!    request that has reached `max_batch_wait_ms` gets the first packing
//!    opportunity. All other requests are stably sorted by cost, then packed
//!    greedily under the cost and count caps. This prevents replenishing
//!    cheap requests from indefinitely overtaking an aged expensive one.
//!    Progress requires a downstream dispatch slot; the wait window is not
//!    an end-to-end latency guarantee.
//! 4. **Timer refresh**: after a partial cost-sorted extract, first/last
//!    timestamps are re-anchored to the requests that remain; once the queue
//!    empties both are cleared so the next submit starts a fresh window.
//! 5. **Runs-alone lane** (long-form audio, #585): items whose
//!    [`HasCost::runs_alone`] is true never share a batch. They are served
//!    one per batch in arrival order, and while batchable items are pending
//!    too the two lanes take turns: a batchable turn, then the oldest flagged
//!    item. A batchable turn serves the batchable items that were pending
//!    when it began, in as many batches as the request count cap and drain
//!    budgets need; items that arrive during the turn wait for the next one.
//!    A turn begins after each flagged item runs, even if no other flagged
//!    item is pending yet, so the items that waited behind it get a turn
//!    before the next one. Before any batch has been taken, the first turn
//!    goes to the lane holding the oldest pending item. Batchable batches
//!    outside turns (none under way, nothing pending flagged, and the last
//!    batch not a flagged item) are packed exactly as point 3.
//!
//!    What a turn takes: it owes the items earlier turns left over, the
//!    oldest whatever its cost, then, oldest first, as many more as fit the
//!    cost cap it began with together. Its budget is that cost cap plus the
//!    oldest left-over item's cost. Besides what it owes, it takes the items
//!    it serves cheapest first, while what is left of its budget still covers
//!    what it owes and the next one fits. Until it has taken an item it does
//!    not owe, it takes the cheapest one even above what is left, as point 2
//!    takes an oversize item. Each batch is packed in cost order under the
//!    cost cap, the request count cap and items' hard caps, a single item
//!    above the cost cap running alone (point 2). Owed items join the batch
//!    at which the next cheapest item would leave too little budget for them,
//!    so a batch stops early only to keep that budget.
//!
//!    Wait bound: every turn serves the oldest left-over item, and no later
//!    arrival is older. So a batchable item is served by the first turn that
//!    begins after it arrives or, if that turn leaves it over, within the
//!    next `n` turns, where `n` counts it and the batchable items that
//!    arrived before it and were still pending when that turn ended: at most
//!    `n + 1` turns in all. Later arrivals never add to `n`, so cheaper items
//!    that keep arriving cannot pass it over turn after turn. A turn takes at
//!    most its cost cap plus two items: the oldest left-over one and the
//!    cheapest other one. A flagged item still waits at most one turn per
//!    flagged item ahead of it, plus one.
//!
//!    The extra budget keeps the share of a turn that goes cheapest first at
//!    no less than the cost cap minus what the other owed items cost, so a
//!    costly left-over item, even one above the cost cap, does not take a
//!    turn from the cheaper ones. Under a load the turns cannot keep up with,
//!    owed items take most of each turn, in arrival order: no item waits
//!    indefinitely and more items are served than in cost order alone, but
//!    the median wait grows, most where the cost cap holds few items (such as
//!    4096 ms, the adaptive floor for a backend that reports no budget).
//!
//! Python currently retains its original first/last timers after a partial
//! extract and clears them only when empty. Rust re-anchors them to the actual
//! remainder because the sidecar's cross-key FCFS selector reads that state:
//! retaining a removed request's timestamp can repeatedly prioritize a hot key
//! and can make a newer tail skip its own coalescing window. Python's packing
//! remains strictly cost-sorted; Rust promotes one aged request per extract
//! (point 3).
//!
//! The runs-alone lane (point 5) is the Python batcher's rule too (#601),
//! with three differences:
//!
//! * What gets flagged: Python flags audio only in the Whisper adapter, at
//!   its feature window, while the sidecar flags every prepared audio item
//!   longer than [`crate::audio_prep::RUNS_ALONE_ABOVE_SAMPLES`], whatever
//!   the model, Parakeet included.
//! * How long a batchable turn lasts: Python's turn is one batch. Its request
//!   count cap is 64, so in practice the cost cap bounds the turn. The
//!   sidecar's count cap is 12 (see [`BatchConfig::default`]), and a turn of
//!   one batch would then leave the costliest pending items behind every
//!   turn, so they could wait indefinitely. The sidecar's turn takes what
//!   Python's batch takes, bounded by the cost cap, split into batches that
//!   respect the count cap.
//! * What a batchable turn serves: Python orders every turn by cost alone,
//!   so cheaper items that keep arriving can pass a costlier one over turn
//!   after turn. The sidecar's turn owes the items earlier turns left over,
//!   which gives the wait bound of point 5.
//!
//! Packing, caps, and flush triggers otherwise follow the Python test
//! contract. The async primitives differ as expected (`tokio::sync` rather
//! than asyncio).

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use tokio::sync::{Mutex, Notify};
use tokio::time::sleep;

use super::batch_config::BatchConfig;

/// Trait for any item that carries a batching cost and an original
/// position within a request.
///
/// Ported from Python's `HasCost` Protocol. `PreparedItem` (built on
/// the Rust-tokenise fast path) implements this directly; test-only
/// stubs in this module do too.
pub trait HasCost {
    /// Batching cost — token count for text, 1 per image, etc.
    fn cost(&self) -> u64;

    /// Optional non-adaptive safety ceiling for a batch containing this item.
    /// Media payloads use this to bound the serialized IPC frame even when the
    /// latency controller raises the model's normal cost target.
    fn hard_batch_cost_cap(&self) -> Option<u64> {
        None
    }

    /// Whether this item must be served in a batch of its own (long-form
    /// audio). Mirrors Python's `PreparedItem.runs_alone`; see point 5 of
    /// the module docs for how such items are ordered.
    fn runs_alone(&self) -> bool {
        false
    }

    /// Original position within the originating request's item list.
    /// Used by the adapter-side output router to zip results back to
    /// the correct request item.
    fn original_index(&self) -> usize;
}

/// A single request waiting to be batched, together with
/// caller-provided routing metadata.
#[derive(Debug, Clone)]
pub struct PendingRequest<I: HasCost, T> {
    pub item: I,
    pub metadata: T,
    pub arrival_time: Instant,
}

/// A flushed batch ready for downstream dispatch.
///
/// Items and metadata are kept in parallel `Vec`s so the adapter-side
/// zip continues to work unchanged.
#[derive(Debug)]
pub struct FormattedBatch<I: HasCost, T> {
    pub items: Vec<I>,
    pub metadata: Vec<T>,
    pub total_cost: u64,
    pub flush_reason: FlushReason,
}

impl<I: HasCost, T> FormattedBatch<I, T> {
    /// Number of items in the batch.
    #[must_use]
    pub fn size(&self) -> usize {
        self.items.len()
    }

    /// Python-parity alias for [`Self::total_cost`] mirroring the
    /// Python `total_tokens` property.
    #[must_use]
    pub fn total_tokens(&self) -> u64 {
        self.total_cost
    }

    /// Return a new batch sorted by cost (ascending). Preserves the
    /// `items[i] ↔ metadata[i]` correspondence.
    ///
    /// Mostly redundant now that `extract_batch` sorts
    /// in place before slicing, but kept for symmetry with the Python
    /// API where some callers sort explicitly.
    #[must_use]
    pub fn sorted_by_cost(self) -> Self
    where
        I: Clone,
        T: Clone,
    {
        // Build index permutation so we don't pay a double-clone on T.
        let mut idx: Vec<usize> = (0..self.items.len()).collect();
        idx.sort_by_key(|&i| self.items[i].cost());

        let items = idx.iter().map(|&i| self.items[i].clone()).collect();
        let metadata = idx.iter().map(|&i| self.metadata[i].clone()).collect();
        Self {
            items,
            metadata,
            total_cost: self.total_cost,
            flush_reason: self.flush_reason,
        }
    }
}

/// Reason a [`BatchFormer`] emitted a batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlushReason {
    CostCap,
    CountCap,
    Timeout,
    Coalesce,
    SingleOversize,
    IdleBypass,
    Drain,
}

impl FlushReason {
    #[must_use]
    pub const fn as_label(self) -> &'static str {
        match self {
            Self::CostCap => "cost_cap",
            Self::CountCap => "count_cap",
            Self::Timeout => "timeout",
            Self::Coalesce => "coalesce",
            Self::SingleOversize => "single_oversize",
            Self::IdleBypass => "idle_bypass",
            Self::Drain => "drain",
        }
    }
}

/// Internal mutable state, guarded by a single [`Mutex`].
///
/// Kept compact on purpose: every batch formation decision reads and
/// mutates this struct, so the lock is hot. No allocations happen
/// under the lock beyond the two parallel `Vec`s.
#[derive(Debug)]
struct Inner<I: HasCost, T> {
    pending: Vec<PendingRequest<I, T>>,
    total_cost: u64,
    first_request_time: Option<Instant>,
    last_submit_time: Option<Instant>,
    /// Lane of the last batch taken: `Some(true)` when it was a runs-alone
    /// item, `None` before the first batch. Decides whose turn is next while
    /// both lanes are pending.
    last_batch_alone: Option<bool>,
    /// The batchable turn under way, if any. See point 5 of the module docs.
    batchable_turn: Option<BatchableTurn>,
    /// Latest cutoff of a batchable turn begun so far. A pending batchable
    /// request that arrived no later than it was left over by an earlier turn.
    last_turn_cutoff: Option<Instant>,
    /// Cached snapshot of the current knobs. Mutations happen via
    /// [`BatchFormer::update_config`] which takes the same lock so
    /// reads of `config` and reads of `pending` stay consistent.
    config: BatchConfig,
}

/// A batchable turn of the runs-alone lane (point 5 of the module docs). It
/// serves the batchable requests that were pending when it began. Of those,
/// it owes the ones earlier turns left over, oldest first, as many as fit its
/// cost cap together and at least the oldest, and serves them whatever their
/// place in cost order; the rest it serves cheapest first while its budget
/// lasts.
#[derive(Debug, Clone, Copy)]
struct BatchableTurn {
    /// Latest arrival among the requests pending when the turn began.
    /// Requests that arrive later wait for the next turn.
    cutoff: Instant,
    /// Latest arrival among the requests the turn owes, while it still owes
    /// any: a request the turn serves that arrived no later than this was
    /// left over by an earlier turn and is owed.
    owed_cutoff: Option<Instant>,
    /// Cost the turn may still take: the cost cap it began with plus the
    /// cost of the oldest request it owes, less what its batches took.
    cost_left: u64,
    /// Whether the turn has taken a request it does not owe. Until it has,
    /// it takes its cheapest such request even above `cost_left`, as the
    /// first batch of a turn takes an oversize item (point 2).
    took_unowed: bool,
}

impl BatchableTurn {
    /// Whether `request` is one of the batchable requests this turn serves.
    fn serves<I: HasCost, T>(&self, request: &PendingRequest<I, T>) -> bool {
        !request.item.runs_alone() && request.arrival_time <= self.cutoff
    }

    /// Whether `request` is one the turn owes.
    fn owes<I: HasCost, T>(&self, request: &PendingRequest<I, T>) -> bool {
        self.serves(request)
            && self
                .owed_cutoff
                .is_some_and(|owed| request.arrival_time <= owed)
    }
}

/// Async batch former.
///
/// Thread-safe for concurrent submits from many tasks. A single
/// consumer task should call [`BatchFormer::get_batch`] in a loop;
/// calling it concurrently from multiple consumers is allowed but has
/// no benefit (and will race over pending items — the consumer is
/// typically the per-model worker loop).
#[derive(Debug)]
pub struct BatchFormer<I: HasCost, T> {
    inner: Mutex<Inner<I, T>>,
    /// Edge-triggered readiness signal. A submit that transitions the
    /// batcher to "first item" or "batch ready" notifies here; the
    /// consumer uses [`Notified::enable`] *inside* the lock to avoid
    /// missed notifications across the drop.
    ready: Notify,
    /// Shared monotonic origin for the lock-free FCFS head timestamp.
    epoch: Instant,
    /// Nanos since `epoch` for the oldest currently-pending item; zero means
    /// empty. Updated while `inner` is locked so queue mutation and FCFS
    /// visibility cannot race.
    fcfs_head_ns: AtomicU64,
    // Generic parameters are consumed by `inner` only; `PhantomData`
    // keeps the type system honest when `Inner` is refactored.
    _phantom: PhantomData<(I, T)>,
}

impl<I: HasCost, T> BatchFormer<I, T> {
    /// Construct a new batch former with the given caps. Use
    /// [`BatchConfig::default`] for Python-parity defaults.
    pub fn new(config: BatchConfig) -> Self {
        Self::with_epoch(config, Instant::now())
    }

    /// Construct with a caller-owned monotonic epoch so FCFS head timestamps
    /// are comparable across sibling batchers.
    pub(crate) fn with_epoch(config: BatchConfig, epoch: Instant) -> Self {
        Self {
            inner: Mutex::new(Inner {
                pending: Vec::new(),
                total_cost: 0,
                first_request_time: None,
                last_submit_time: None,
                last_batch_alone: None,
                batchable_turn: None,
                last_turn_cutoff: None,
                config,
            }),
            ready: Notify::new(),
            epoch,
            fcfs_head_ns: AtomicU64::new(0),
            _phantom: PhantomData,
        }
    }

    /// Lock-free FCFS head snapshot. Zero means this batcher is empty.
    pub(crate) fn fcfs_head_ns(&self) -> u64 {
        self.fcfs_head_ns.load(Ordering::Acquire)
    }

    /// Current cap snapshot. Returned by value — the caller sees a
    /// consistent view even if another task updates the config
    /// concurrently.
    pub async fn config(&self) -> BatchConfig {
        self.inner.lock().await.config
    }

    /// Update the caps atomically. The adaptive controller calls this
    /// after each `step()`.
    pub async fn update_config(&self, config: BatchConfig) {
        let mut guard = self.inner.lock().await;
        guard.config = config;
        // Config shrinks can make the pending queue instantly
        // yieldable (e.g. new `max_batch_cost` < current `total_cost`
        // or new `max_batch_requests` < `pending.len()`). Wake the
        // consumer so it re-evaluates the flush triggers.
        if should_yield_batch(&guard) {
            self.ready.notify_one();
        }
    }

    /// Number of pending items. Cheap snapshot under the lock.
    pub async fn pending_count(&self) -> usize {
        self.inner.lock().await.pending.len()
    }

    /// Total cost across all pending items.
    pub async fn pending_cost(&self) -> u64 {
        self.inner.lock().await.total_cost
    }

    /// Python-parity alias for [`Self::pending_cost`].
    pub async fn pending_tokens(&self) -> u64 {
        self.pending_cost().await
    }

    /// Snapshot of `first_request_time`. Used by the FCFS fairness
    /// primitive (`scheduler::pick_next_lora`) to sort per-LoRA
    /// batchers by oldest head.
    ///
    /// Python accessed `_first_request_time` directly (private field);
    /// the Rust scheduler formalises that coupling as a public method.
    pub async fn first_request_time(&self) -> Option<Instant> {
        self.inner.lock().await.first_request_time
    }

    /// Enqueue a single item.
    pub async fn submit(&self, item: I, metadata: T) {
        let mut guard = self.inner.lock().await;
        let was_empty = guard.pending.is_empty();
        let notify = append_item(&mut guard, item, metadata);
        if was_empty {
            self.refresh_fcfs_head(&guard);
        }
        drop(guard);
        if notify {
            self.ready.notify_one();
        }
    }

    /// Enqueue many items under a single lock acquisition. More
    /// efficient than looping over [`Self::submit`] for multi-item
    /// requests (each external request can carry up to 64 items).
    pub async fn submit_many(&self, items: Vec<(I, T)>) {
        if items.is_empty() {
            return;
        }
        let mut guard = self.inner.lock().await;
        let was_empty = guard.pending.is_empty();
        let mut notify = false;
        for (item, metadata) in items {
            notify |= append_item(&mut guard, item, metadata);
        }
        if was_empty {
            self.refresh_fcfs_head(&guard);
        }
        drop(guard);
        if notify {
            self.ready.notify_one();
        }
    }

    /// Block until the next batch is ready, then return it.
    ///
    /// Readiness is one of: cost cap hit, count cap hit, timeout
    /// elapsed since first item, or coalesce window elapsed since
    /// last submit.
    ///
    /// Setting `immediate = true` short-circuits the wait: if any
    /// items are pending, they flush immediately. Used by the worker
    /// loop after an idle period to eliminate an unnecessary batch
    /// wait at low concurrency.
    pub async fn get_batch(&self, immediate: bool) -> FormattedBatch<I, T> {
        loop {
            // Arm the notification *before* we drop the lock so a
            // submit that fires between "decide to wait" and "start
            // waiting" is not missed. See the Tokio docs on
            // `Notified::enable` for the pattern.
            let notified = self.ready.notified();
            tokio::pin!(notified);

            let wait = {
                let mut guard = self.inner.lock().await;
                if let Some(reason) = flush_reason(&guard, immediate) {
                    let batch = extract_batch(&mut guard, reason);
                    self.refresh_fcfs_head(&guard);
                    return batch;
                }
                notified.as_mut().enable();
                wait_timeout(&guard)
            };

            match wait {
                None => {
                    // No pending items → wait indefinitely for the
                    // first submit.
                    notified.await;
                }
                Some(dur) if dur.is_zero() => {
                    // The timer is already expired — loop and flush
                    // on the next lock acquisition. This path fires
                    // when `get_batch` is called with `immediate=false`
                    // and the timers have already elapsed (e.g. a
                    // slow consumer catching up).
                    continue;
                }
                Some(dur) => {
                    tokio::select! {
                        biased;
                        () = &mut notified => {}
                        () = sleep(dur) => {}
                    }
                }
            }
        }
    }

    /// Non-blocking: return a batch if the flush conditions already
    /// hold, else return `None`. Matches Python's `try_get_batch`.
    pub async fn try_get_batch(&self) -> Option<FormattedBatch<I, T>> {
        let mut guard = self.inner.lock().await;
        let reason = flush_reason(&guard, false)?;
        let batch = extract_batch(&mut guard, reason);
        self.refresh_fcfs_head(&guard);
        Some(batch)
    }

    /// Drain whatever is pending right now, bypassing the flush
    /// conditions. Matches Python's `try_drain` — the continuous
    /// batching tail called after each forward pass.
    pub async fn try_drain(&self) -> Option<FormattedBatch<I, T>> {
        self.try_drain_up_to(usize::MAX).await
    }

    /// Drain at most `max_items`, bypassing flush conditions while still
    /// honouring the configured cost/request caps.
    ///
    /// The scheduler drain loop uses this to bound one continuous-batching
    /// wave to the backlog count observed when its first continuation slot
    /// opens. A newer cheap item can participate ahead of an unaged expensive
    /// item; the oldest request gets the first packing opportunity once its
    /// wait window expires. The count budget bounds wave size but does not
    /// identify an exact request cohort.
    pub async fn try_drain_up_to(&self, max_items: usize) -> Option<FormattedBatch<I, T>> {
        if max_items == 0 {
            return None;
        }
        let mut guard = self.inner.lock().await;
        if guard.pending.is_empty() {
            None
        } else {
            let batch = extract_batch_up_to(&mut guard, FlushReason::Drain, max_items);
            self.refresh_fcfs_head(&guard);
            Some(batch)
        }
    }

    /// Refresh the atomic FCFS head from the queue while `inner` is locked.
    ///
    /// [`extract_batch_up_to`] recomputes the timers from the actual remainder,
    /// so `first_request_time` is also the FCFS head after every mutation.
    fn refresh_fcfs_head(&self, inner: &Inner<I, T>) {
        let head = inner
            .first_request_time
            .map(|instant| self.timestamp_ns(instant))
            .unwrap_or(0);
        self.fcfs_head_ns.store(head, Ordering::Release);
    }

    fn timestamp_ns(&self, instant: Instant) -> u64 {
        let nanos = instant
            .saturating_duration_since(self.epoch)
            .as_nanos()
            .min(u128::from(u64::MAX)) as u64;
        // Zero is reserved for an empty batcher.
        nanos.max(1)
    }
}

// ---------------------------------------------------------------------
// Free-standing helpers. Kept outside the `BatchFormer` impl so the
// hot-path decisions are easy to test in isolation (see the `tests`
// module below) and so they don't borrow `self` implicitly.
// ---------------------------------------------------------------------

fn append_item<I: HasCost, T>(inner: &mut Inner<I, T>, item: I, metadata: T) -> bool {
    let cost = item.cost();
    let now = Instant::now();

    inner.pending.push(PendingRequest {
        item,
        metadata,
        arrival_time: now,
    });
    inner.total_cost += cost;
    inner.last_submit_time = Some(now);

    let is_first = inner.first_request_time.is_none();
    if is_first {
        inner.first_request_time = Some(now);
    }

    // Wake the consumer if the batch is now full, OR if this is the
    // very first item (so the consumer can start the timeout window
    // instead of blocking on an indefinite `notified()`).
    should_yield_batch(inner) || is_first
}

fn batch_timeout_expired<I: HasCost, T>(inner: &Inner<I, T>) -> bool {
    let Some(first) = inner.first_request_time else {
        return false;
    };
    let elapsed_ms = first.elapsed().as_secs_f64() * 1000.0;
    elapsed_ms >= inner.config.max_batch_wait_ms
}

fn coalesce_expired<I: HasCost, T>(inner: &Inner<I, T>) -> bool {
    let Some(last) = inner.last_submit_time else {
        return false;
    };
    if inner.pending.is_empty() {
        return false;
    }
    let elapsed_ms = last.elapsed().as_secs_f64() * 1000.0;
    elapsed_ms >= inner.config.effective_coalesce_ms()
}

fn should_yield_batch<I: HasCost, T>(inner: &Inner<I, T>) -> bool {
    flush_reason(inner, false).is_some()
}

fn effective_max_cost<I: HasCost, T>(inner: &Inner<I, T>) -> u64 {
    inner
        .pending
        .iter()
        .filter_map(|request| request.item.hard_batch_cost_cap())
        .fold(inner.config.max_batch_cost, u64::min)
}

fn flush_reason<I: HasCost, T>(inner: &Inner<I, T>, immediate: bool) -> Option<FlushReason> {
    if inner.pending.is_empty() {
        return None;
    }
    let max_cost = effective_max_cost(inner);
    if inner.pending.len() == 1 && inner.total_cost > max_cost {
        return Some(FlushReason::SingleOversize);
    }
    if inner.total_cost >= max_cost {
        return Some(FlushReason::CostCap);
    }
    if inner.pending.len() >= inner.config.max_batch_requests {
        return Some(FlushReason::CountCap);
    }
    if batch_timeout_expired(inner) {
        return Some(FlushReason::Timeout);
    }
    if coalesce_expired(inner) {
        return Some(FlushReason::Coalesce);
    }
    if immediate {
        return Some(FlushReason::IdleBypass);
    }
    None
}

/// Time remaining until the next guaranteed flush, given current
/// state. `None` means "wait indefinitely" (no pending items).
fn wait_timeout<I: HasCost, T>(inner: &Inner<I, T>) -> Option<Duration> {
    let first = inner.first_request_time?;
    let now = Instant::now();

    let elapsed_ms = now.saturating_duration_since(first).as_secs_f64() * 1000.0;
    let batch_remaining_ms = (inner.config.max_batch_wait_ms - elapsed_ms).max(0.0);

    let coalesce_ms = inner.config.effective_coalesce_ms();
    let coalesce_remaining_ms = match inner.last_submit_time {
        Some(last) => {
            let since_last = now.saturating_duration_since(last).as_secs_f64() * 1000.0;
            (coalesce_ms - since_last).max(0.0)
        }
        None => coalesce_ms,
    };

    let effective_ms = batch_remaining_ms.min(coalesce_remaining_ms);
    // ms → Duration, clamping tiny negatives to zero.
    Some(Duration::try_from_secs_f64(effective_ms.max(0.0) / 1000.0).unwrap_or(Duration::MAX))
}

/// Decide the lane of the next batch, keeping the batchable turn in step,
/// and order pending requests so that packing takes the batch from the front.
/// Returns `true` when the next batch is a runs-alone item.
///
/// Lane decision (point 5 of the module docs): a batchable turn under way
/// goes on while it owes a request, while a request it serves still fits
/// what is left of its budget, or, until it has taken a request it does not
/// owe, while it serves any. Otherwise, while both lanes are pending, the
/// lane that did not take the last batch goes next, and before the first
/// batch the lane holding the oldest pending request. A batchable turn opens
/// with the first batchable batch after a runs-alone item ran, even if no
/// other runs-alone item is pending yet, and with the first batchable batch
/// while runs-alone items are pending.
///
/// Order: batchable requests first, those the turn serves ahead of later
/// arrivals, each group sorted by cost; runs-alone requests after them in
/// arrival order, the oldest moved to the front when it is its turn. The sort
/// is stable, so FIFO order holds within equal costs (Python's `list.sort` is
/// stable too). Then [`make_room_for_owed`] moves the requests the turn owes
/// forward where packing would otherwise leave no room for them. Outside
/// batchable turns, with no runs-alone request pending, this is point 3: the
/// oldest request that has waited `max_batch_wait_ms` first, the rest by
/// cost. Mirrors Python's
/// `BatchFormer._order_runs_alone_lane`, except that a batchable turn can
/// span several batches and serves the requests earlier turns left over.
///
/// `max_cost` is the batch cost cap before the turn's own budget, and the
/// cost cap a turn opened now begins with; `max_take` is the most requests
/// the batch may take.
fn order_for_next_batch<I: HasCost, T>(
    inner: &mut Inner<I, T>,
    max_cost: u64,
    max_take: usize,
) -> bool {
    let alone_pending = inner.pending.iter().any(|r| r.item.runs_alone());
    let batchable_pending = inner.pending.iter().any(|r| !r.item.runs_alone());
    if let Some(turn) = inner.batchable_turn {
        // Owed requests always fit (see `make_room_for_owed`).
        let goes_on = |r: &PendingRequest<I, T>| {
            turn.serves(r) && (!turn.took_unowed || r.item.cost() <= turn.cost_left)
        };
        if !inner.pending.iter().any(goes_on) {
            inner.batchable_turn = None;
        }
    }
    let alone_next = alone_pending
        && (!batchable_pending
            || (inner.batchable_turn.is_none()
                && match inner.last_batch_alone {
                    Some(last_alone) => !last_alone,
                    None => inner
                        .pending
                        .iter()
                        .min_by_key(|r| r.arrival_time)
                        .is_some_and(|oldest| oldest.item.runs_alone()),
                }));
    if !alone_next
        && inner.batchable_turn.is_none()
        && (alone_pending || inner.last_batch_alone == Some(true))
    {
        let cutoff = inner
            .pending
            .iter()
            .map(|r| r.arrival_time)
            .max()
            .expect("batchable requests are pending");
        let (owed_cutoff, oldest_cost) =
            owed_left_over(&inner.pending, inner.last_turn_cutoff, max_cost);
        inner.batchable_turn = Some(BatchableTurn {
            cutoff,
            owed_cutoff,
            cost_left: max_cost.saturating_add(oldest_cost),
            took_unowed: false,
        });
        inner.last_turn_cutoff = inner.last_turn_cutoff.max(Some(cutoff));
    }

    // Outside batchable turns, with no runs-alone request pending, packing is
    // point 3's: a timeout must give the oldest request an extraction
    // opportunity even while cheaper arrivals keep the queue full. Inside a
    // turn, what the turn owes bounds the wait instead (point 5). Select
    // before sorting so tied arrival timestamps retain their pending order.
    let oldest_aged = if inner.batchable_turn.is_none() && !alone_pending {
        inner
            .pending
            .iter()
            .enumerate()
            .min_by_key(|(_, request)| request.arrival_time)
            .filter(|(_, request)| {
                request.arrival_time.elapsed().as_secs_f64() * 1000.0
                    >= inner.config.max_batch_wait_ms
            })
            .map(|(index, _)| index)
    } else {
        None
    };
    let promoted = oldest_aged.map(|index| inner.pending.remove(index));

    // Cost order keeps each batch's items close in length, minimising
    // padding waste on the adapter side.
    let turn = inner.batchable_turn;
    let served = |r: &PendingRequest<I, T>| turn.is_some_and(|t| t.serves(r));
    inner.pending.sort_by(|a, b| {
        let alone = a.item.runs_alone();
        alone
            .cmp(&b.item.runs_alone())
            .then_with(|| served(b).cmp(&served(a)))
            .then_with(|| {
                if alone {
                    a.arrival_time.cmp(&b.arrival_time)
                } else {
                    a.item.cost().cmp(&b.item.cost())
                }
            })
    });
    if let Some(request) = promoted {
        inner.pending.insert(0, request);
    }
    if alone_next {
        if let Some(head) = inner.pending.iter().position(|r| r.item.runs_alone()) {
            inner.pending[..=head].rotate_right(1);
        }
    } else if let Some(turn) = inner
        .batchable_turn
        .as_mut()
        .filter(|t| t.owed_cutoff.is_some())
    {
        let room = max_cost.min(turn.cost_left);
        if !make_room_for_owed(&mut inner.pending, turn, room, max_take) {
            turn.owed_cutoff = None;
        }
    }
    alone_next
}

/// The requests a turn opened now owes: of those that arrived no later than
/// `carried_cutoff` (left over by earlier turns), the oldest and, oldest
/// first, as many more as fit `max_cost` together with them. Requests that
/// arrived at the same instant are owed together. Returns the latest arrival
/// among the owed requests and the cost of the oldest ones, which the turn
/// adds to its budget.
fn owed_left_over<I: HasCost, T>(
    pending: &[PendingRequest<I, T>],
    carried_cutoff: Option<Instant>,
    max_cost: u64,
) -> (Option<Instant>, u64) {
    let Some(carried_cutoff) = carried_cutoff else {
        return (None, 0);
    };
    // A min-heap by arrival: only the owed few are popped, so a long queue
    // costs one linear pass rather than a sort.
    let mut left_over: BinaryHeap<Reverse<(Instant, u64)>> = pending
        .iter()
        .filter(|r| !r.item.runs_alone() && r.arrival_time <= carried_cutoff)
        .map(|r| Reverse((r.arrival_time, r.item.cost())))
        .collect();
    // Pops the next group of requests that arrived at the same instant.
    let mut next_group = || {
        let Reverse((arrival, mut cost)) = left_over.pop()?;
        while let Some(&Reverse((next, next_cost))) = left_over.peek() {
            if next != arrival {
                break;
            }
            cost += next_cost;
            left_over.pop();
        }
        Some((arrival, cost))
    };
    let Some((mut owed_cutoff, oldest_cost)) = next_group() else {
        return (None, 0);
    };
    let mut owed_cost = oldest_cost;
    while let Some((arrival, cost)) = next_group() {
        if owed_cost + cost > max_cost {
            break;
        }
        owed_cost += cost;
        owed_cutoff = arrival;
    }
    (Some(owed_cutoff), oldest_cost)
}

/// Makes sure packing leaves room for the requests `turn` owes. `pending`
/// holds the requests the turn serves first, sorted by cost. Packing takes
/// them in that order while each next one still leaves what is left of the
/// turn's budget enough for the owed requests not yet taken; at the first
/// one that would not, the owed requests move ahead of it, still in cost
/// order, and go next. The batch therefore stays in cost order, a request the
/// turn owes is never left over again, and packing stops only where the
/// cost cap `room`, the request count `max_take` or what the turn owes stops
/// it. Returns whether the turn still owes a pending request.
fn make_room_for_owed<I: HasCost, T>(
    pending: &mut [PendingRequest<I, T>],
    turn: &BatchableTurn,
    room: u64,
    max_take: usize,
) -> bool {
    let mut owes_any = false;
    let mut owed_left: u64 = 0;
    for request in pending.iter().filter(|r| turn.owes(r)) {
        owes_any = true;
        owed_left += request.item.cost();
    }
    if !owes_any {
        return false;
    }
    let served = pending.iter().take_while(|r| turn.serves(r)).count();
    let mut batch_cost: u64 = 0;
    for (count, at) in (0..served).enumerate() {
        let request = &pending[at];
        let cost = request.item.cost();
        if count >= max_take || (batch_cost + cost > room && count > 0) {
            break;
        }
        if turn.owes(request) {
            owed_left -= cost;
        } else if batch_cost + cost + owed_left > turn.cost_left {
            pending[at..served].sort_by_key(|r| !turn.owes(r));
            break;
        }
        batch_cost += cost;
    }
    true
}

fn extract_batch<I: HasCost, T>(
    inner: &mut Inner<I, T>,
    flush_reason: FlushReason,
) -> FormattedBatch<I, T> {
    extract_batch_up_to(inner, flush_reason, usize::MAX)
}

fn extract_batch_up_to<I: HasCost, T>(
    inner: &mut Inner<I, T>,
    flush_reason: FlushReason,
    max_items: usize,
) -> FormattedBatch<I, T> {
    debug_assert!(max_items > 0, "zero-item drains are rejected by the caller");
    let pending_before = inner.pending.len();
    let pending_alone = inner.pending.iter().filter(|r| r.item.runs_alone()).count();
    let max_cost = effective_max_cost(inner);
    let max_requests = inner.config.max_batch_requests;
    let max_take = max_items.min(max_requests).max(1);
    let alone_next = order_for_next_batch(inner, max_cost, max_take);

    // A batch of a batchable turn takes only requests the turn serves, within
    // what is left of the turn's budget.
    let turn = if alone_next {
        None
    } else {
        inner.batchable_turn
    };
    let max_cost = turn.map_or(max_cost, |t| max_cost.min(t.cost_left));

    // Greedy pack. Always take at least one item so a single large
    // request (cost ≥ max_cost) doesn't starve.
    let mut batch_cost: u64 = 0;
    let mut take_count: usize = 0;
    for req in &inner.pending {
        if take_count >= max_items {
            break;
        }
        let runs_alone = req.item.runs_alone();
        // A runs-alone item never shares a batch: stop in front of one
        // unless it is the first item taken.
        if runs_alone && take_count > 0 {
            break;
        }
        // Requests that arrived during a batchable turn wait for the next.
        if turn.is_some_and(|t| !t.serves(req)) && take_count > 0 {
            break;
        }
        let c = req.item.cost();
        // Stop if taking this item would exceed the cost cap — but
        // only if we already have at least one item in the batch.
        // Otherwise (take_count == 0) fall through and take a single
        // oversize item alone: Python does the same, to avoid starving
        // a request whose cost is simply larger than the cap.
        if batch_cost + c > max_cost && take_count > 0 {
            break;
        }
        batch_cost += c;
        take_count += 1;
        if runs_alone || take_count >= max_requests {
            break;
        }
    }
    let alone = take_count > 0 && inner.pending[0].item.runs_alone();
    if take_count > 0 {
        // Whose turn is next when both lanes are pending.
        inner.last_batch_alone = Some(alone);
        if alone {
            inner.batchable_turn = None;
        } else if let Some(turn) = inner.batchable_turn.as_mut() {
            let snapshot = *turn;
            turn.took_unowed |= inner.pending[..take_count]
                .iter()
                .any(|r| !snapshot.owes(r));
            turn.cost_left = turn.cost_left.saturating_sub(batch_cost);
        }
    }

    // Drain the taken prefix into parallel vecs.
    let drained: Vec<PendingRequest<I, T>> = inner.pending.drain(..take_count).collect();
    inner.total_cost -= batch_cost;
    tracing::debug!(
        flush_reason = flush_reason.as_label(),
        lane = if alone { "alone" } else { "batched" },
        batch_size = take_count,
        total_cost = batch_cost,
        pending = pending_before,
        pending_alone,
        remaining = inner.pending.len(),
        "scheduler batch formed",
    );

    let mut items = Vec::with_capacity(drained.len());
    let mut metadata = Vec::with_capacity(drained.len());
    for req in drained {
        items.push(req.item);
        metadata.push(req.metadata);
    }

    // Re-anchor both timers to the requests that actually remain. Cost-sorted
    // extraction can remove the oldest or newest arrival from the middle of
    // the original FIFO order; retaining either removed timestamp would make a
    // newer tail flush prematurely or wait against the wrong coalesce window.
    let mut first = None;
    let mut last = None;
    for request in &inner.pending {
        let arrival = request.arrival_time;
        first = Some(first.map_or(arrival, |current: Instant| current.min(arrival)));
        last = Some(last.map_or(arrival, |current: Instant| current.max(arrival)));
    }
    inner.first_request_time = first;
    inner.last_submit_time = last;

    FormattedBatch {
        items,
        metadata,
        total_cost: batch_cost,
        flush_reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::time::{sleep as tsleep, Duration as TDuration};

    /// Minimal `HasCost` stub: carries a fixed cost and a position.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct StubItem {
        cost: u64,
        idx: usize,
    }

    impl StubItem {
        fn new(cost: u64) -> Self {
            Self { cost, idx: 0 }
        }

        fn with_idx(cost: u64, idx: usize) -> Self {
            Self { cost, idx }
        }
    }

    impl HasCost for StubItem {
        fn cost(&self) -> u64 {
            self.cost
        }
        fn original_index(&self) -> usize {
            self.idx
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct HardCapItem {
        cost: u64,
        cap: u64,
    }

    impl HasCost for HardCapItem {
        fn cost(&self) -> u64 {
            self.cost
        }
        fn hard_batch_cost_cap(&self) -> Option<u64> {
            Some(self.cap)
        }
        fn original_index(&self) -> usize {
            0
        }
    }

    // Arrival age is a std::Instant value, so these tests backdate the real
    // pending requests rather than advancing Tokio's separate virtual clock.
    async fn set_request_ages<I: HasCost>(
        former: &BatchFormer<I, usize>,
        ages_ms: &[(usize, u64)],
    ) {
        let now = Instant::now();
        let mut guard = former.inner.lock().await;
        for request in &mut guard.pending {
            let age_ms = ages_ms
                .iter()
                .find(|(metadata, _)| *metadata == request.metadata)
                .expect("every pending request has an explicit age")
                .1;
            request.arrival_time = now - Duration::from_millis(age_ms);
        }
        guard.first_request_time = guard.pending.iter().map(|r| r.arrival_time).min();
        guard.last_submit_time = guard.pending.iter().map(|r| r.arrival_time).max();
        former.refresh_fcfs_head(&guard);
    }

    #[tokio::test]
    async fn aged_requests_progress_in_arrival_order_under_replenishment() {
        let former = BatchFormer::<StubItem, usize>::new(BatchConfig {
            max_batch_cost: 10,
            max_batch_requests: 8,
            max_batch_wait_ms: 20.0,
            ..BatchConfig::default()
        });
        // Pending order and cost order both differ from actual arrival order.
        for id in [2, 0, 1] {
            former.submit(StubItem::with_idx(10, id), id).await;
        }
        former.submit(StubItem::with_idx(1, 3), 3).await;
        set_request_ages(&former, &[(2, 100), (0, 300), (1, 200), (3, 0)]).await;

        for id in 0..3 {
            let batch = former.try_drain_up_to(1).await.unwrap();
            assert_eq!(batch.metadata, vec![id], "oldest aged request goes first");
            assert_eq!(batch.items, vec![StubItem::with_idx(10, id)]);
            assert_eq!(batch.total_cost, 10);
            former.submit(StubItem::with_idx(1, id + 4), id + 4).await;
        }
        assert_eq!(former.pending_count().await, 4);
        assert_eq!(former.pending_cost().await, 4);
        let remainder = former.try_drain().await.unwrap();
        assert_eq!(remainder.metadata, vec![3, 4, 5, 6]);
        assert_eq!(former.pending_count().await, 0);
        assert_eq!(former.pending_cost().await, 0);
        assert_eq!(former.fcfs_head_ns(), 0);
    }

    #[tokio::test]
    async fn aged_oversize_request_flushes_alone_among_smaller_requests() {
        let former = BatchFormer::<StubItem, usize>::new(BatchConfig {
            max_batch_cost: 10,
            max_batch_requests: 8,
            max_batch_wait_ms: 20.0,
            ..BatchConfig::default()
        });
        former.submit(StubItem::with_idx(20, 0), 0).await;
        former.submit(StubItem::with_idx(0, 1), 1).await;
        former.submit(StubItem::with_idx(1, 2), 2).await;
        set_request_ages(&former, &[(0, 100), (1, 0), (2, 0)]).await;
        let remaining_first = former.inner.lock().await.pending[1].arrival_time;

        let batch = former.try_get_batch().await.unwrap();
        assert_eq!(batch.metadata, vec![0]);
        assert_eq!(batch.items, vec![StubItem::with_idx(20, 0)]);
        assert_eq!(batch.total_cost, 20);
        assert_eq!(former.pending_count().await, 2);
        assert_eq!(former.pending_cost().await, 1);
        let guard = former.inner.lock().await;
        assert_eq!(guard.first_request_time, Some(remaining_first));
        assert_eq!(guard.last_submit_time, Some(remaining_first));
        assert_ne!(former.fcfs_head_ns(), 0);
    }

    #[tokio::test]
    async fn aged_promotion_preserves_hard_cost_and_updated_count_caps() {
        let config = BatchConfig {
            max_batch_cost: 1_000,
            max_batch_requests: 8,
            max_batch_wait_ms: 20.0,
            ..BatchConfig::default()
        };
        let former = BatchFormer::<HardCapItem, usize>::new(config);
        for (id, cost) in [4, 2, 2, 2].into_iter().enumerate() {
            former.submit(HardCapItem { cost, cap: 7 }, id).await;
        }
        set_request_ages(&former, &[(0, 100), (1, 0), (2, 0), (3, 0)]).await;

        let batch = former.try_drain_up_to(3).await.unwrap();
        assert_eq!(batch.metadata, vec![0, 1]);
        assert_eq!(
            batch.total_cost, 6,
            "promotion must still honor the hard cap"
        );
        assert_eq!(former.pending_cost().await, 4);
        former
            .update_config(BatchConfig {
                max_batch_cost: 5,
                max_batch_requests: 1,
                ..config
            })
            .await;
        let batch = former.try_drain_up_to(3).await.unwrap();
        assert_eq!(batch.metadata, vec![2]);
        assert_eq!(batch.total_cost, 2);
        assert_eq!(former.pending_count().await, 1);
    }

    #[tokio::test]
    async fn promotion_tracks_updated_wait_threshold() {
        let config = BatchConfig {
            max_batch_cost: 10,
            max_batch_requests: 8,
            max_batch_wait_ms: 1_000_000.0,
            ..BatchConfig::default()
        };
        let former = BatchFormer::<StubItem, usize>::new(config);
        former.submit(StubItem::with_idx(10, 0), 0).await;
        former.submit(StubItem::with_idx(1, 1), 1).await;
        set_request_ages(&former, &[(0, 100), (1, 0)]).await;
        let first = former.try_drain_up_to(1).await.unwrap();
        assert_eq!(
            first.metadata,
            vec![1],
            "unaged requests remain cost-sorted"
        );
        former.submit(StubItem::with_idx(1, 2), 2).await;
        former
            .update_config(BatchConfig {
                max_batch_wait_ms: 20.0,
                ..config
            })
            .await;

        let next = former.get_batch(true).await;
        assert_eq!(next.metadata, vec![0]);
        assert_eq!(next.total_cost, 10);
        assert_eq!(former.pending_count().await, 1);
        assert_eq!(former.pending_cost().await, 1);
    }

    #[tokio::test]
    async fn submit_and_drain_roundtrip() {
        let b = BatchFormer::<StubItem, u32>::new(BatchConfig::default());
        b.submit(StubItem::new(10), 1).await;
        b.submit(StubItem::new(20), 2).await;
        assert_eq!(b.pending_count().await, 2);
        assert_eq!(b.pending_cost().await, 30);

        let batch = b.try_drain().await.expect("drain should succeed");
        assert_eq!(batch.size(), 2);
        assert_eq!(batch.total_cost, 30);
        assert_eq!(batch.flush_reason, FlushReason::Drain);
        assert_eq!(b.pending_count().await, 0);
        assert_eq!(b.pending_cost().await, 0);
    }

    #[tokio::test]
    async fn bounded_drain_takes_only_snapshot_budget() {
        let b = BatchFormer::<StubItem, usize>::new(BatchConfig {
            max_batch_cost: 1_000_000,
            max_batch_requests: 64,
            ..BatchConfig::default()
        });
        for idx in 0..5 {
            b.submit(StubItem::with_idx(1, idx), idx).await;
        }

        assert!(b.try_drain_up_to(0).await.is_none());
        let first = b
            .try_drain_up_to(2)
            .await
            .expect("bounded drain should return pending work");
        assert_eq!(first.metadata, vec![0, 1]);
        assert_eq!(first.size(), 2);
        assert_eq!(b.pending_count().await, 3);

        let second = b
            .try_drain_up_to(2)
            .await
            .expect("second bounded drain should preserve the same cap");
        assert_eq!(second.metadata, vec![2, 3]);
        assert_eq!(b.pending_count().await, 1);
    }

    #[tokio::test]
    async fn bounded_drain_caps_count_not_exact_arrival_cohort() {
        let b = BatchFormer::<StubItem, usize>::new(BatchConfig {
            max_batch_cost: 1_000_000,
            max_batch_requests: 64,
            ..BatchConfig::default()
        });
        b.submit(StubItem::with_idx(100, 0), 0).await;
        let snapshot_budget = b.pending_count().await;
        b.submit(StubItem::with_idx(1, 1), 1).await;

        let drained = b
            .try_drain_up_to(snapshot_budget)
            .await
            .expect("bounded drain should return one item");
        assert_eq!(drained.metadata, vec![1], "packing remains cost-sorted");
        assert_eq!(drained.size(), snapshot_budget);
        assert_eq!(b.pending_count().await, 1);
        assert_ne!(
            b.fcfs_head_ns(),
            0,
            "the displaced older item remains visible to FCFS"
        );
    }

    #[tokio::test]
    async fn partial_extract_reanchors_timers_to_remaining_items() {
        let b = BatchFormer::<StubItem, usize>::new(BatchConfig::default());
        b.submit(StubItem::with_idx(1, 0), 0).await;
        b.submit(StubItem::with_idx(1, 1), 1).await;
        let expected_remaining_arrival = {
            let guard = b.inner.lock().await;
            guard.pending[1].arrival_time
        };

        let drained = b
            .try_drain_up_to(1)
            .await
            .expect("oldest equal-cost item should drain first");
        assert_eq!(drained.metadata, vec![0]);
        let guard = b.inner.lock().await;
        assert_eq!(guard.pending.len(), 1);
        assert_eq!(guard.first_request_time, Some(expected_remaining_arrival));
        assert_eq!(guard.last_submit_time, Some(expected_remaining_arrival));
        assert_ne!(
            b.fcfs_head_ns(),
            0,
            "the remaining item must stay visible to cross-key FCFS"
        );
    }

    #[tokio::test]
    async fn cost_cap_triggers_extract() {
        let cfg = BatchConfig {
            max_batch_cost: 30,
            max_batch_requests: 64,
            max_batch_wait_ms: 10_000.0, // big so only cost triggers
            coalesce_ms: 10_000.0,
            coalesce_ratio: 1.0,
        };
        let b = BatchFormer::<StubItem, u32>::new(cfg);
        b.submit(StubItem::new(15), 1).await;
        b.submit(StubItem::new(15), 2).await;
        // total_cost (30) >= max_batch_cost (30) → should_yield now.
        let batch = b.try_get_batch().await.expect("cost cap should trigger");
        assert_eq!(batch.size(), 2);
        assert_eq!(batch.total_cost, 30);
        assert_eq!(batch.flush_reason, FlushReason::CostCap);
    }

    #[tokio::test]
    async fn item_hard_cap_overrides_larger_adaptive_cost_cap() {
        let cfg = BatchConfig {
            max_batch_cost: 1_000_000,
            max_batch_requests: 64,
            max_batch_wait_ms: 10_000.0,
            coalesce_ms: 10_000.0,
            coalesce_ratio: 1.0,
        };
        let b = BatchFormer::<HardCapItem, u32>::new(cfg);
        b.submit(
            HardCapItem {
                cost: 400_000,
                cap: 720_000,
            },
            1,
        )
        .await;
        b.submit(
            HardCapItem {
                cost: 400_000,
                cap: 720_000,
            },
            2,
        )
        .await;

        let batch = b.try_get_batch().await.expect("hard cap should trigger");
        assert_eq!(batch.size(), 1);
        assert_eq!(batch.total_cost, 400_000);
        assert_eq!(batch.flush_reason, FlushReason::CostCap);
        assert_eq!(b.pending_count().await, 1);
    }

    #[tokio::test]
    async fn count_cap_triggers_extract() {
        let cfg = BatchConfig {
            max_batch_cost: 1_000_000,
            max_batch_requests: 3,
            max_batch_wait_ms: 10_000.0,
            coalesce_ms: 10_000.0,
            coalesce_ratio: 1.0,
        };
        let b = BatchFormer::<StubItem, u32>::new(cfg);
        b.submit(StubItem::new(1), 1).await;
        b.submit(StubItem::new(1), 2).await;
        b.submit(StubItem::new(1), 3).await;
        let batch = b.try_get_batch().await.expect("count cap should trigger");
        assert_eq!(batch.size(), 3);
        assert_eq!(batch.flush_reason, FlushReason::CountCap);
    }

    #[tokio::test]
    async fn cost_sort_reorders_pending_ascending() {
        let cfg = BatchConfig {
            max_batch_cost: 1_000_000,
            max_batch_requests: 64,
            max_batch_wait_ms: 1_000_000.0,
            ..BatchConfig::default()
        };
        let b = BatchFormer::<StubItem, usize>::new(cfg);
        // Insert out-of-order costs.
        b.submit(StubItem::with_idx(50, 0), 0).await;
        b.submit(StubItem::with_idx(10, 1), 1).await;
        b.submit(StubItem::with_idx(30, 2), 2).await;
        b.submit(StubItem::with_idx(10, 3), 3).await;

        let batch = b.try_drain().await.unwrap();
        let costs: Vec<u64> = batch.items.iter().map(|i| i.cost()).collect();
        assert_eq!(costs, vec![10, 10, 30, 50], "must be sorted ascending");
        assert_eq!(
            batch.metadata,
            vec![1, 3, 2, 0],
            "equal-cost FIFO is stable"
        );
    }

    #[tokio::test]
    async fn oversize_item_flushes_alone() {
        let cfg = BatchConfig {
            max_batch_cost: 100,
            max_batch_requests: 64,
            ..BatchConfig::default()
        };
        let b = BatchFormer::<StubItem, u32>::new(cfg);
        b.submit(StubItem::new(500), 1).await; // cost >> cap

        let batch = b
            .try_get_batch()
            .await
            .expect("oversize should still flush");
        assert_eq!(batch.size(), 1);
        assert_eq!(batch.total_cost, 500);
        assert_eq!(batch.flush_reason, FlushReason::SingleOversize);
    }

    #[tokio::test]
    async fn extract_leaves_remainder_in_queue() {
        let cfg = BatchConfig {
            max_batch_cost: 25,
            max_batch_requests: 64,
            ..BatchConfig::default()
        };
        let b = BatchFormer::<StubItem, u32>::new(cfg);
        b.submit(StubItem::new(10), 1).await;
        b.submit(StubItem::new(10), 2).await;
        b.submit(StubItem::new(10), 3).await;
        // total_cost = 30 ≥ 25 → should_yield; extract packs two (sum 20),
        // drops the third because 20+10=30 > 25.
        let batch = b.try_get_batch().await.unwrap();
        assert_eq!(batch.size(), 2);
        assert_eq!(batch.total_cost, 20);
        assert_eq!(b.pending_count().await, 1);
        assert_eq!(b.pending_cost().await, 10);
    }

    #[tokio::test]
    async fn empty_batcher_returns_none_from_try_paths() {
        let b = BatchFormer::<StubItem, u32>::new(BatchConfig::default());
        assert!(b.try_get_batch().await.is_none());
        assert!(b.try_drain().await.is_none());
    }

    #[tokio::test]
    async fn immediate_get_batch_flushes_without_timeout() {
        let cfg = BatchConfig {
            max_batch_wait_ms: 10_000.0, // would block for 10 seconds otherwise
            coalesce_ms: 10_000.0,
            coalesce_ratio: 1.0,
            ..BatchConfig::default()
        };
        let b = BatchFormer::<StubItem, u32>::new(cfg);
        b.submit(StubItem::new(1), 1).await;

        // This would hang without `immediate = true`; with it, it
        // returns as soon as the single pending item is visible.
        let batch = b.get_batch(true).await;
        assert_eq!(batch.size(), 1);
        assert_eq!(batch.flush_reason, FlushReason::IdleBypass);
    }

    #[tokio::test]
    async fn timers_reset_when_queue_empties() {
        let b = BatchFormer::<StubItem, u32>::new(BatchConfig::default());
        b.submit(StubItem::new(1), 1).await;
        assert!(b.first_request_time().await.is_some());
        let _ = b.try_drain().await.unwrap();
        assert!(
            b.first_request_time().await.is_none(),
            "emptying the queue must reset the first-request timer"
        );
    }

    #[tokio::test]
    async fn get_batch_times_out_after_max_wait() {
        // Real-clock test: `wait_timeout` reads `std::Instant::now`,
        // so mixing tokio's pause() with wall time causes spin loops.
        // A real 40 ms ceiling keeps the test fast enough to run in
        // the hot suite.
        let cfg = BatchConfig {
            max_batch_cost: 1_000_000,
            max_batch_requests: 1000,
            max_batch_wait_ms: 20.0,
            coalesce_ms: 1_000.0,
            coalesce_ratio: 1.0, // coalesce ties max-wait; timeout wins by priority
        };
        let b = BatchFormer::<StubItem, u32>::new(cfg);
        b.submit(StubItem::new(1), 1).await;

        // Expect get_batch to complete between the wait window and
        // the timeout ceiling.
        let start = std::time::Instant::now();
        let batch = tokio::time::timeout(TDuration::from_millis(200), b.get_batch(false))
            .await
            .expect("must flush on timeout within 200 ms");
        let elapsed_ms = start.elapsed().as_millis();
        assert_eq!(batch.size(), 1);
        assert_eq!(batch.flush_reason, FlushReason::Timeout);
        assert!(
            elapsed_ms >= 15,
            "must wait at least ~max_batch_wait_ms before flushing (waited {elapsed_ms} ms)"
        );
    }

    #[tokio::test]
    async fn submit_many_takes_one_lock() {
        // Functional check: all items end up pending in arrival order.
        let b = BatchFormer::<StubItem, u32>::new(BatchConfig::default());
        let items: Vec<(StubItem, u32)> = (0..5)
            .map(|i| (StubItem::with_idx(i as u64 + 1, i), i as u32))
            .collect();
        b.submit_many(items).await;
        assert_eq!(b.pending_count().await, 5);
        assert_eq!(b.pending_cost().await, 1 + 2 + 3 + 4 + 5);
    }

    #[tokio::test]
    async fn update_config_wakes_consumer_when_new_cap_already_satisfied() {
        // NB: don't pause() here — the consumer races a real submit
        // through Notify, and the pattern is intrinsically clock-free
        // once the Notified future is enabled inside the lock.
        let original = BatchConfig {
            max_batch_cost: 1_000_000,
            max_batch_requests: 1000,
            max_batch_wait_ms: 1_000_000.0,
            coalesce_ms: 1_000_000.0,
            coalesce_ratio: 1.0,
        };
        let b = std::sync::Arc::new(BatchFormer::<StubItem, u32>::new(original));
        b.submit(StubItem::new(5), 1).await;

        // Spawn a consumer that will block in `get_batch`.
        let consumer = {
            let b = std::sync::Arc::clone(&b);
            tokio::spawn(async move { b.get_batch(false).await })
        };

        // Yield so the consumer can reach the await point.
        tokio::task::yield_now().await;
        tsleep(TDuration::from_millis(1)).await;

        // Shrink the cap so the one pending item now trips
        // `batch_is_full`; the update must wake the consumer.
        //
        // Build the new config from a plain copy of `original` so we
        // don't re-acquire the inner lock while `update_config` is
        // about to acquire it — that would deadlock.
        let shrunk = BatchConfig {
            max_batch_cost: 1,
            max_batch_requests: 1,
            ..original
        };
        b.update_config(shrunk).await;

        let batch = tokio::time::timeout(TDuration::from_secs(2), consumer)
            .await
            .expect("consumer must finish within 2s of the config shrink")
            .unwrap();
        assert_eq!(batch.size(), 1);
    }

    #[test]
    fn wait_timeout_shrinks_as_time_passes() {
        // Build an `Inner` directly so we can test the sync helper.
        let cfg = BatchConfig {
            max_batch_wait_ms: 20.0,
            coalesce_ms: 1_000_000.0,
            coalesce_ratio: 1.0,
            ..BatchConfig::default()
        };
        let now = Instant::now();
        let inner: Inner<StubItem, u32> = Inner {
            pending: vec![PendingRequest {
                item: StubItem::new(1),
                metadata: 1,
                arrival_time: now,
            }],
            total_cost: 1,
            first_request_time: Some(now - Duration::from_millis(5)),
            last_submit_time: Some(now),
            last_batch_alone: None,
            batchable_turn: None,
            last_turn_cutoff: None,
            config: cfg,
        };
        let t = wait_timeout(&inner).expect("pending → Some");
        // roughly 15 ms remaining (20 - 5), ±5 ms slop for clock jitter.
        assert!(t <= Duration::from_millis(20));
    }

    #[test]
    fn wait_timeout_saturates_for_windows_beyond_duration_range() {
        let cfg = BatchConfig {
            max_batch_wait_ms: 1e300,
            coalesce_ms: 1e300,
            coalesce_ratio: 1.0,
            ..BatchConfig::default()
        };
        let now = Instant::now();
        let inner: Inner<StubItem, u32> = Inner {
            pending: vec![PendingRequest {
                item: StubItem::new(1),
                metadata: 1,
                arrival_time: now,
            }],
            total_cost: 1,
            first_request_time: Some(now),
            last_submit_time: Some(now),
            last_batch_alone: None,
            batchable_turn: None,
            last_turn_cutoff: None,
            config: cfg,
        };
        assert_eq!(wait_timeout(&inner), Some(Duration::MAX));
    }

    #[test]
    fn wait_timeout_none_when_no_pending() {
        let cfg = BatchConfig::default();
        let inner: Inner<StubItem, u32> = Inner {
            pending: vec![],
            total_cost: 0,
            first_request_time: None,
            last_submit_time: None,
            last_batch_alone: None,
            batchable_turn: None,
            last_turn_cutoff: None,
            config: cfg,
        };
        assert!(wait_timeout(&inner).is_none());
    }

    #[test]
    fn coalesce_clamps_to_proportional_under_short_wait() {
        // Coalesce window = min(15, 4 * 0.5) = 2 ms.
        let cfg = BatchConfig {
            coalesce_ms: 15.0,
            coalesce_ratio: 0.5,
            max_batch_wait_ms: 4.0,
            ..BatchConfig::default()
        };
        assert!((cfg.effective_coalesce_ms() - 2.0).abs() < f64::EPSILON);
    }

    // ---- Runs-alone lane (long-form audio, #585) ----

    /// Audio stub: cost is duration in ms and the hard cap is the audio batch
    /// cap, as `SchedulerItem` sets them for prepared audio. `alone` marks
    /// long-form audio.
    #[derive(Debug, Clone, Copy)]
    struct LaneItem {
        cost: u64,
        alone: bool,
        hard_cap: Option<u64>,
    }

    impl HasCost for LaneItem {
        fn cost(&self) -> u64 {
            self.cost
        }
        fn hard_batch_cost_cap(&self) -> Option<u64> {
            self.hard_cap
        }
        fn runs_alone(&self) -> bool {
            self.alone
        }
        fn original_index(&self) -> usize {
            0
        }
    }

    fn clip_ms(cost: u64) -> LaneItem {
        LaneItem {
            cost,
            alone: false,
            hard_cap: Some(crate::audio_prep::MAX_AUDIO_BATCH_DURATION_MS),
        }
    }

    fn long_form_ms(cost: u64) -> LaneItem {
        LaneItem {
            alone: true,
            ..clip_ms(cost)
        }
    }

    fn clip(seconds: u64) -> LaneItem {
        clip_ms(seconds * 1_000)
    }

    fn long_form(seconds: u64) -> LaneItem {
        long_form_ms(seconds * 1_000)
    }

    /// A batcher queue driven with explicit arrival times, in ms after `epoch`.
    struct Lanes {
        inner: Inner<LaneItem, String>,
        epoch: Instant,
    }

    impl Lanes {
        fn new(max_batch_cost: u64) -> Self {
            Self {
                inner: Inner {
                    pending: Vec::new(),
                    total_cost: 0,
                    first_request_time: None,
                    last_submit_time: None,
                    last_batch_alone: None,
                    batchable_turn: None,
                    last_turn_cutoff: None,
                    config: BatchConfig {
                        max_batch_cost,
                        // Arrival times here are simulated, while point 3's
                        // aging reads the real clock: keep it out of reach
                        // unless a test lowers it.
                        max_batch_wait_ms: f64::INFINITY,
                        ..BatchConfig::default()
                    },
                },
                epoch: Instant::now(),
            }
        }

        /// Queue `item` as if it arrived `at_ms` after the epoch.
        fn arrive(&mut self, item: LaneItem, name: &str, at_ms: u64) {
            append_item(&mut self.inner, item, name.to_string());
            let request = self.inner.pending.last_mut().expect("just appended");
            request.arrival_time = self.epoch + Duration::from_millis(at_ms);
        }

        fn take_up_to(&mut self, max_items: usize) -> FormattedBatch<LaneItem, String> {
            extract_batch_up_to(&mut self.inner, FlushReason::Drain, max_items)
        }

        fn take(&mut self) -> Vec<String> {
            self.take_up_to(usize::MAX).metadata
        }

        fn pending_names(&self) -> Vec<String> {
            self.inner
                .pending
                .iter()
                .map(|request| request.metadata.clone())
                .collect()
        }
    }

    /// SplitMix64: reproducible test data without a `rand` dependency.
    struct TestRng(u64);

    impl TestRng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        /// Uniform in `low..=high`.
        fn range(&mut self, low: u64, high: u64) -> u64 {
            low + self.next_u64() % (high - low + 1)
        }

        fn chance(&mut self, percent: u64) -> bool {
            self.range(0, 99) < percent
        }

        fn pick<T: Copy>(&mut self, choices: &[T]) -> T {
            choices[self.range(0, choices.len() as u64 - 1) as usize]
        }
    }

    /// Cost-sorted greedy packing exactly as `extract_batch_up_to` did before
    /// the runs-alone lane. `pending` holds `(cost, hard_cap, name)` in queue
    /// order and is updated in place, as the batcher updates its own queue.
    fn previous_packing(
        pending: &mut Vec<(u64, Option<u64>, String)>,
        config: &BatchConfig,
        max_items: usize,
    ) -> (Vec<String>, u64) {
        pending.sort_by_key(|(cost, _, _)| *cost);
        let max_cost = pending
            .iter()
            .filter_map(|(_, cap, _)| *cap)
            .fold(config.max_batch_cost, u64::min);
        let mut batch_cost = 0;
        let mut take_count = 0;
        for (cost, _, _) in pending.iter() {
            if take_count >= max_items {
                break;
            }
            if batch_cost + cost > max_cost && take_count > 0 {
                break;
            }
            batch_cost += cost;
            take_count += 1;
            if take_count >= config.max_batch_requests {
                break;
            }
        }
        let taken = pending.drain(..take_count).map(|(_, _, name)| name);
        (taken.collect(), batch_cost)
    }

    #[test]
    fn unflagged_batches_match_previous_cost_sorted_packing() {
        for seed in 0..20 {
            let mut rng = TestRng(seed);
            let mut lanes = Lanes::new(rng.range(1, 4_000));
            lanes.inner.config.max_batch_requests = rng.range(1, 8) as usize;
            let config = lanes.inner.config;
            let mut reference: Vec<(u64, Option<u64>, String)> = Vec::new();
            let mut now_ms = 0;
            let mut next_id = 0;

            for _ in 0..60 {
                for _ in 0..rng.range(0, 5) {
                    now_ms += rng.range(0, 3);
                    let cost = rng.range(1, 1_500);
                    let hard_cap = rng.chance(20).then(|| rng.range(500, 4_000));
                    let name = format!("item-{next_id}");
                    next_id += 1;
                    let item = LaneItem {
                        cost,
                        alone: false,
                        hard_cap,
                    };
                    lanes.arrive(item, &name, now_ms);
                    reference.push((cost, hard_cap, name));
                }
                if reference.is_empty() {
                    continue;
                }
                let max_items = rng.pick(&[usize::MAX, usize::MAX, 1, 2, 5]);

                let batch = lanes.take_up_to(max_items);

                let (expected, expected_cost) =
                    previous_packing(&mut reference, &config, max_items);
                assert_eq!(batch.metadata, expected, "seed {seed}");
                assert_eq!(batch.total_cost, expected_cost, "seed {seed}");
                let remaining: Vec<&String> = reference.iter().map(|(_, _, name)| name).collect();
                assert_eq!(
                    lanes.pending_names().iter().collect::<Vec<_>>(),
                    remaining,
                    "seed {seed}"
                );
                let remaining_cost: u64 = reference.iter().map(|(cost, _, _)| cost).sum();
                assert_eq!(lanes.inner.total_cost, remaining_cost, "seed {seed}");
            }
        }
    }

    #[test]
    fn long_form_runs_alone_after_clips_that_arrived_around_it() {
        // Clips are not held in a long recording's batch (head-of-line wait).
        let mut lanes = Lanes::new(720_000);
        lanes.arrive(clip(5), "clip-5s", 0);
        lanes.arrive(long_form(240), "long-240s", 1_000);
        lanes.arrive(clip(8), "clip-8s", 2_000);

        assert_eq!(lanes.take(), ["clip-5s", "clip-8s"]);
        assert_eq!(lanes.take(), ["long-240s"]);
        assert!(lanes.inner.pending.is_empty());
        assert_eq!(lanes.inner.total_cost, 0);
    }

    #[test]
    fn long_form_items_run_in_arrival_order() {
        // An older long recording is not starved by a newer, shorter one.
        let mut lanes = Lanes::new(720_000);
        lanes.arrive(long_form(250), "long-250s", 0);
        lanes.arrive(long_form(190), "long-190s", 1_000);

        assert_eq!(lanes.take(), ["long-250s"]);
        assert_eq!(lanes.take(), ["long-190s"]);
    }

    #[test]
    fn first_batch_comes_from_the_lane_of_the_oldest_request() {
        for (long_arrival_ms, order) in [(0, ["long", "clip"]), (2_000, ["clip", "long"])] {
            let mut lanes = Lanes::new(720_000);
            lanes.arrive(clip(5), "clip", 1_000);
            lanes.arrive(long_form(240), "long", long_arrival_ms);

            let first = lanes.take();
            let second = lanes.take();
            assert_eq!([first, second].concat(), order);
        }
    }

    #[test]
    fn lanes_take_turns_while_both_are_pending() {
        // Two long recordings and clips that keep arriving: clips, long, clips, long.
        let mut lanes = Lanes::new(720_000);
        lanes.arrive(clip(4), "clip-a", 0);
        lanes.arrive(clip(6), "clip-b", 1_000);
        lanes.arrive(long_form(240), "long-1", 2_000);
        lanes.arrive(long_form(200), "long-2", 3_000);

        let mut served = Vec::new();
        for second in 4..9 {
            served.push(lanes.take());
            // One clip lands while each batch runs.
            lanes.arrive(clip(5), &format!("clip@{second}"), second * 1_000);
        }

        assert_eq!(
            served,
            [
                vec!["clip-a", "clip-b"],
                vec!["long-1"],
                vec!["clip@4", "clip@5"],
                vec!["long-2"],
                vec!["clip@6", "clip@7"],
            ]
        );
        assert_eq!(lanes.pending_names(), ["clip@8"]);
    }

    /// What a closed-loop run served: per client, how many requests were
    /// served and the most clip turns one of them waited. A request waits one
    /// turn when the first clip turn that begins after it arrives serves it.
    struct ClosedLoopTurns {
        served: Vec<usize>,
        worst_wait_turns: Vec<usize>,
        clip_turns: usize,
    }

    /// Extracts `batches` batches from `lanes`, whose pending requests are
    /// named by client index; every served client resubmits its item at once.
    /// Clip turns are counted as runs of clip batches, so this needs a long
    /// recording pending throughout: then every clip turn is followed by one.
    fn closed_loop_turns(
        lanes: &mut Lanes,
        clients: &[LaneItem],
        batches: usize,
        now_ms: &mut u64,
    ) -> ClosedLoopTurns {
        let mut run = ClosedLoopTurns {
            served: vec![0; clients.len()],
            worst_wait_turns: vec![0; clients.len()],
            clip_turns: 0,
        };
        // Clip turns begun before each pending client's request arrived.
        let mut turns_at_submit = vec![0; clients.len()];
        let mut last_alone = None;
        for _ in 0..batches {
            assert!(
                lanes.inner.pending.iter().any(|r| r.item.alone),
                "a long recording stays pending"
            );
            let batch = lanes.take_up_to(usize::MAX);
            let alone = batch.items[0].alone;
            if !alone && last_alone != Some(false) {
                run.clip_turns += 1;
            }
            last_alone = Some(alone);
            *now_ms += 1;
            for name in &batch.metadata {
                let client: usize = name.parse().expect("client index");
                run.served[client] += 1;
                if !alone {
                    let waited = run.clip_turns - turns_at_submit[client];
                    run.worst_wait_turns[client] = run.worst_wait_turns[client].max(waited);
                }
                turns_at_submit[client] = run.clip_turns;
                lanes.arrive(clients[client], name, *now_ms);
            }
        }
        // Requests still pending have waited at least the turns begun since.
        for request in &lanes.inner.pending {
            let client: usize = request.metadata.parse().expect("client index");
            let waited = run.clip_turns - turns_at_submit[client];
            run.worst_wait_turns[client] = run.worst_wait_turns[client].max(waited);
        }
        run
    }

    #[test]
    fn clip_left_over_by_a_turn_is_served_by_the_next_turn() {
        // At a speech model's adaptive floor (180000 ms), 18 clips of 10 s
        // fill a clip turn's cost cap, and their clients resubmit at once. The
        // 30 s clip, costliest, is left over by its first turn. The next turn
        // owes it, so the 10 s clips that arrived since cannot pass it over
        // again; that turn's budget grows by its 30 s, so it still serves all
        // 18 of them. A long recording still runs after every clip turn.
        let mut lanes = Lanes::new(180_000);
        let mut clients = vec![long_form_ms(240_000), clip_ms(30_000)];
        clients.extend([clip_ms(10_000); 18]);
        let mut now_ms = 0;
        for (client, item) in clients.iter().enumerate() {
            lanes.arrive(*item, &client.to_string(), now_ms);
            now_ms += 1;
        }

        let run = closed_loop_turns(&mut lanes, &clients, 300, &mut now_ms);

        // One recording, then 100 rounds of a clip turn of two batches and a
        // recording, the last round cut short by the 300-batch horizon.
        assert_eq!(run.clip_turns, 100);
        assert_eq!(run.served[0], 100, "a recording runs after every clip turn");
        assert_eq!(run.worst_wait_turns[0], 1);
        // The 30 s clip's client resubmits during a turn, so each request has
        // a first turn that leaves it over and is served by the next.
        assert_eq!(run.served[1], 50, "the 30 s clip is served every 2 turns");
        assert_eq!(run.worst_wait_turns[1], 2);
        for client in 2..clients.len() {
            assert_eq!(run.served[client], 100, "10 s clip client {client}");
            assert_eq!(run.worst_wait_turns[client], 1, "10 s clip client {client}");
        }
    }

    #[test]
    fn turn_batches_stay_in_cost_order_and_full_around_owed_clips() {
        // Two costly clips are left over by a turn. The next turn owes the
        // older one (L1); the other (L2) does not fit the cost cap with it,
        // so it is served cheapest first like the newer 5 s clips. Its first
        // batch takes the 5 s clips and L2, in cost order, and stops only at
        // the cost cap; L1 follows in the same turn, before the next
        // recording. Arrival-order leaders would instead stop the first batch
        // at L2 and take three batches.
        let mut lanes = Lanes::new(180_000);
        lanes.arrive(long_form(240), "r0", 0);
        lanes.arrive(clip_ms(100_000), "L1", 1);
        lanes.arrive(clip_ms(90_000), "L2", 2);
        lanes.arrive(long_form(240), "r1", 3);
        lanes.arrive(long_form(240), "r2", 4);
        for i in 0..9 {
            lanes.arrive(clip(20), &format!("m{i}"), 5 + i);
        }
        assert_eq!(lanes.take(), ["r0"]);
        let turn_1: Vec<String> = (0..9).map(|i| format!("m{i}")).collect();
        assert_eq!(lanes.take(), turn_1, "20 s clips fill the cost cap");
        assert_eq!(lanes.take(), ["r1"]);
        for i in 0..10 {
            lanes.arrive(clip(5), &format!("s{i}"), 100 + i);
        }

        let mut first: Vec<String> = (0..10).map(|i| format!("s{i}")).collect();
        first.push("L2".into());
        assert_eq!(lanes.take(), first);
        assert_eq!(lanes.take(), ["L1"]);
        assert_eq!(lanes.take(), ["r2"]);
        assert!(lanes.inner.pending.is_empty());
    }

    #[test]
    fn oversize_left_over_clip_runs_alone_beside_the_turn() {
        // At 16384 ms (the cost cap for a backend that reports no budget) a
        // 20 s clip runs alone. Left over, it is owed by the next turn, whose
        // budget grows by its cost, so the 2 s clips still get the turn's
        // whole cost cap instead of losing the turn to it.
        let mut lanes = Lanes::new(16_384);
        lanes.arrive(long_form(240), "r0", 0);
        lanes.arrive(clip(20), "big", 1);
        lanes.arrive(long_form(240), "r1", 2);
        for i in 0..8 {
            lanes.arrive(clip(2), &format!("a{i}"), 3 + i);
        }
        assert_eq!(lanes.take(), ["r0"]);
        let turn_1: Vec<String> = (0..8).map(|i| format!("a{i}")).collect();
        assert_eq!(lanes.take(), turn_1, "8 clips of 2 s fit 16384 ms");
        assert_eq!(lanes.take(), ["r1"]);
        lanes.arrive(long_form(240), "r2", 50);
        for i in 0..8 {
            lanes.arrive(clip(2), &format!("b{i}"), 100 + i);
        }

        let turn_2: Vec<String> = (0..8).map(|i| format!("b{i}")).collect();
        assert_eq!(lanes.take(), turn_2, "the cheaper clips keep the cost cap");
        assert_eq!(lanes.take(), ["big"], "the owed clip runs alone");
        assert_eq!(lanes.take(), ["r2"]);
    }

    #[test]
    fn clip_left_over_before_the_last_recording_is_owed_after_it() {
        // The batch after the last recording opens a turn although nothing
        // flagged is pending any more, so a clip left over before it is still
        // owed: cheaper clips that arrive during that turn wait for the next
        // one, while plain cost-sorted packing (point 3) would take them
        // ahead of it.
        let mut lanes = Lanes::new(180_000);
        lanes.arrive(clip(30), "x", 0);
        for i in 0..18 {
            lanes.arrive(clip(10), &format!("c{i}"), 1 + i);
        }
        lanes.arrive(long_form(240), "r", 100);
        assert_eq!(lanes.take().len(), 12);
        assert_eq!(lanes.take().len(), 6);
        assert_eq!(lanes.pending_names(), ["x", "r"]);
        assert_eq!(lanes.take(), ["r"]);

        for i in 0..12 {
            lanes.arrive(clip(1), &format!("n{i}"), 200 + i);
        }
        let n: Vec<String> = (0..12).map(|i| format!("n{i}")).collect();
        assert_eq!(lanes.take(), n, "the request count cap splits the turn");
        for i in 0..12 {
            lanes.arrive(clip(1), &format!("k{i}"), 300 + i);
        }
        assert_eq!(lanes.take(), ["x"], "the turn still owes x");
        let k: Vec<String> = (0..12).map(|i| format!("k{i}")).collect();
        assert_eq!(lanes.take(), k);
    }

    #[test]
    fn long_form_amid_steady_clips_waits_for_one_clip_batch() {
        // Clip traffic that never lets up delays a flagged recording just
        // over 30 s by one clip batch, not for as long as clips keep coming.
        let mut lanes = Lanes::new(720_000);
        lanes.arrive(long_form(240), "long-240s", 0);
        assert_eq!(lanes.take(), ["long-240s"]);

        // A clip and a 31 s recording land while long-240s runs, and one more
        // clip lands while each later batch runs.
        lanes.arrive(clip(4), "clip@1", 1_000);
        lanes.arrive(long_form(31), "long-31s", 2_000);
        let mut served = Vec::new();
        for second in 3..7 {
            lanes.arrive(clip(4), &format!("clip@{second}"), second * 1_000);
            served.push(lanes.take());
        }

        assert_eq!(
            served,
            [
                vec!["clip@1", "clip@3"],
                vec!["long-31s"],
                vec!["clip@4", "clip@5"],
                vec!["clip@6"],
            ]
        );
    }

    #[tokio::test]
    async fn try_drain_takes_turns_and_keeps_counts_consistent() {
        let b = BatchFormer::<LaneItem, &str>::new(BatchConfig {
            max_batch_cost: 720_000,
            ..BatchConfig::default()
        });
        b.submit_many(vec![
            (clip(5), "clip-5s"),
            (long_form(240), "long-240s"),
            (clip(8), "clip-8s"),
            (long_form(200), "long-200s"),
            (clip(3), "clip-3s"),
        ])
        .await;

        // A drain budget of one item cuts the first clip turn short; the
        // turn goes on with the clips it began with before long-240s runs,
        // while a clip that lands meanwhile waits for the next turn.
        let expected: [(usize, &[&str], usize, u64); 5] = [
            (1, &["clip-3s"], 4, 453_000),
            (4, &["clip-5s", "clip-8s"], 3, 441_000),
            (2, &["long-240s"], 2, 201_000),
            (usize::MAX, &["clip-1s"], 1, 200_000),
            (usize::MAX, &["long-200s"], 0, 0),
        ];
        for (step, (max_items, names, pending_count, pending_cost)) in
            expected.into_iter().enumerate()
        {
            if step == 1 {
                tsleep(TDuration::from_millis(2)).await;
                b.submit(clip(1), "clip-1s").await;
            }
            let batch = b.try_drain_up_to(max_items).await.expect("work is pending");
            assert_eq!(batch.metadata, names, "step {step}");
            assert_eq!(
                batch.total_cost,
                batch.items.iter().map(|item| item.cost).sum::<u64>()
            );
            assert_eq!(b.pending_count().await, pending_count, "step {step}");
            assert_eq!(b.pending_cost().await, pending_cost, "step {step}");
        }
        assert!(b.try_drain().await.is_none());
        assert!(b.first_request_time().await.is_none());
        assert_eq!(b.fcfs_head_ns(), 0);
    }

    #[tokio::test]
    async fn runs_alone_batch_keeps_flush_reason_and_reanchors_timers() {
        let b = BatchFormer::<LaneItem, &str>::new(BatchConfig {
            max_batch_cost: 16_384,
            max_batch_wait_ms: 10_000.0,
            coalesce_ms: 10_000.0,
            coalesce_ratio: 1.0,
            ..BatchConfig::default()
        });
        b.submit(long_form(240), "long").await;
        // Keep the two arrivals distinct so the long recording is the oldest.
        tsleep(TDuration::from_millis(2)).await;
        b.submit(clip(5), "clip").await;
        let clip_arrival = b.inner.lock().await.pending[1].arrival_time;

        // 245 s pending reaches the cost cap; the oldest request's lane goes first.
        let batch = b.try_get_batch().await.expect("cost cap should trigger");
        assert_eq!(batch.metadata, ["long"]);
        assert_eq!(batch.flush_reason, FlushReason::CostCap);
        {
            let guard = b.inner.lock().await;
            assert_eq!(guard.first_request_time, Some(clip_arrival));
            assert_eq!(guard.last_submit_time, Some(clip_arrival));
        }
        assert_ne!(b.fcfs_head_ns(), 0, "the clip stays visible to FCFS");
        assert!(
            b.try_get_batch().await.is_none(),
            "the clip waits for its own window"
        );
    }

    /// What a closed-loop run measured: completion latencies, in ms, of clips
    /// and of long recordings. Requests still pending at the end count as
    /// waiting until then.
    struct ClosedLoopLatency {
        clips: Vec<u64>,
        recordings: Vec<u64>,
    }

    impl ClosedLoopLatency {
        fn worst_clip_ms(&self) -> u64 {
            self.clips.iter().copied().max().unwrap_or(0)
        }

        fn worst_recording_ms(&self) -> u64 {
            self.recordings.iter().copied().max().unwrap_or(0)
        }

        fn median_clip_ms(&self) -> u64 {
            let mut clips = self.clips.clone();
            clips.sort_unstable();
            clips[clips.len() / 2]
        }
    }

    /// Closed-loop clients keep one request each in flight on one speech
    /// model served one batch at a time with the cost cap `max_batch_cost`,
    /// for 30 minutes.
    ///
    /// Clips are 2-30 s and mostly short; recordings are 240 s and take
    /// 11.3 s of GPU time each; a clip batch takes 0.3 s plus 0.12 s per
    /// clip. Some extracts are cut short by a drain's item budget, as the
    /// dispatcher's continuation waves do.
    fn closed_loop_latency(
        clip_clients: usize,
        long_clients: usize,
        seed: u64,
        max_batch_cost: u64,
    ) -> ClosedLoopLatency {
        const HORIZON_MS: u64 = 1_800_000;
        let mut rng = TestRng(seed);
        let mut lanes = Lanes::new(max_batch_cost);
        let total = clip_clients + long_clients;
        // Next submit time per client; a client is idle while its request is pending.
        let mut next_submit: Vec<Option<u64>> =
            (0..total).map(|client| Some(client as u64)).collect();
        let mut submitted_at = vec![0u64; total];
        let mut run = ClosedLoopLatency {
            clips: Vec::new(),
            recordings: Vec::new(),
        };
        let mut now_ms = 0u64;
        while now_ms < HORIZON_MS {
            for client in 0..total {
                if next_submit[client].is_some_and(|at| at <= now_ms) {
                    let at = next_submit[client].take().expect("checked above");
                    let item = if client < clip_clients {
                        let seconds = match rng.range(0, 99) {
                            0..=79 => rng.range(2, 10),
                            80..=92 => rng.range(10, 20),
                            _ => rng.range(20, 30),
                        };
                        clip(seconds)
                    } else {
                        long_form(240)
                    };
                    lanes.arrive(item, &client.to_string(), at);
                    submitted_at[client] = at;
                }
            }
            if lanes.inner.pending.is_empty() {
                now_ms = next_submit
                    .iter()
                    .flatten()
                    .copied()
                    .min()
                    .expect("a client is idle");
                continue;
            }
            let max_items = rng.pick(&[usize::MAX, usize::MAX, usize::MAX, 1, 3]);
            let batch = lanes.take_up_to(max_items);
            now_ms += if batch.items[0].alone {
                11_300
            } else {
                300 + 120 * batch.size() as u64
            };
            for name in &batch.metadata {
                let client: usize = name.parse().expect("client index");
                let latency = now_ms - submitted_at[client];
                if client < clip_clients {
                    run.clips.push(latency);
                } else {
                    run.recordings.push(latency);
                }
                next_submit[client] = Some(now_ms + rng.range(1, 50));
            }
        }
        for request in &lanes.inner.pending {
            let client: usize = request.metadata.parse().expect("client index");
            let waited = now_ms - submitted_at[client];
            if client < clip_clients {
                run.clips.push(waited);
            } else {
                run.recordings.push(waited);
            }
        }
        run
    }

    #[test]
    fn closed_loop_clips_and_recordings_are_served_within_a_few_turns() {
        // More clips pend at a clip turn than the request count cap (12)
        // takes in one batch. A turn still serves every clip that was
        // pending when it began, so the longest clips are not passed over
        // turn after turn, and recordings keep their turns.
        for (clip_clients, long_clients) in [(16, 4), (16, 1), (32, 4), (48, 4)] {
            for seed in 0..3 {
                let run = closed_loop_latency(clip_clients, long_clients, seed, 720_000);
                let case =
                    format!("{clip_clients} clip / {long_clients} long clients, seed {seed}");
                let (clip_ms, long_ms) = (run.worst_clip_ms(), run.worst_recording_ms());
                assert!(clip_ms < 60_000, "{case}: a clip waited {clip_ms} ms");
                assert!(long_ms < 120_000, "{case}: a recording waited {long_ms} ms");
            }
        }
    }

    #[test]
    fn closed_loop_clips_are_served_at_the_adaptive_floor_too() {
        // At a speech model's adaptive floor (180000 ms) a clip turn takes a
        // quarter of what it takes at 720000 ms, so a turn leaves clips over
        // whenever those pending add up to more than 180 s. The next turn
        // owes them, so the costliest clips are not passed over turn after
        // turn by the clips their clients resubmit. Without that, with 32 or
        // more clip clients a clip waited most of the 30-minute run.
        for (clip_clients, long_clients) in [(16, 4), (16, 1), (32, 4), (48, 4)] {
            for seed in 0..3 {
                let run = closed_loop_latency(clip_clients, long_clients, seed, 180_000);
                let case =
                    format!("{clip_clients} clip / {long_clients} long clients, seed {seed}");
                let (clip_ms, long_ms) = (run.worst_clip_ms(), run.worst_recording_ms());
                assert!(clip_ms < 90_000, "{case}: a clip waited {clip_ms} ms");
                assert!(long_ms < 120_000, "{case}: a recording waited {long_ms} ms");
            }
        }
    }

    #[test]
    fn closed_loop_at_small_cost_caps_bounds_waits_and_keeps_short_clips_quick() {
        // A cost cap below the longer clips: 16384 ms, the cap for a backend
        // that reports no budget, and 4096 ms, its adaptive floor. Clips
        // above the cap run alone, and the load is more than the lanes can
        // serve. Cost-first packing alone then served the costliest clips
        // only when nothing cheaper was pending, so some waited the whole
        // 30-minute run. Serving every left-over clip first instead served
        // clips in about arrival order, so most waited minutes. Owing the
        // oldest left-over clips, with the turn's budget raised by the
        // oldest one's cost, bounds every wait and keeps cheaper clips first
        // otherwise: at 16384 ms the median clip stays within seconds of
        // what cost-first packing gives.
        for (clip_clients, long_clients) in [(16, 4), (32, 4), (48, 4)] {
            for seed in 0..3 {
                let case =
                    format!("{clip_clients} clip / {long_clients} long clients, seed {seed}");
                let run = closed_loop_latency(clip_clients, long_clients, seed, 16_384);
                let (worst, median) = (run.worst_clip_ms(), run.median_clip_ms());
                assert!(
                    worst < 600_000,
                    "{case}, 16384 ms: a clip waited {worst} ms"
                );
                assert!(median < 60_000, "{case}, 16384 ms: median clip {median} ms");
                assert!(
                    run.clips.len() > 450,
                    "{case}, 16384 ms: {} clips",
                    run.clips.len()
                );

                let run = closed_loop_latency(clip_clients, long_clients, seed, 4_096);
                let worst = run.worst_clip_ms();
                assert!(worst < 900_000, "{case}, 4096 ms: a clip waited {worst} ms");
                assert!(
                    run.clips.len() > 280,
                    "{case}, 4096 ms: {} clips",
                    run.clips.len()
                );
            }
        }
    }

    /// The batchable turn the audit sees under way.
    struct AuditTurn {
        /// Turns begun so far, this one included.
        index: usize,
        /// The clips it may still serve, by name with their cost.
        clips: std::collections::HashMap<String, u64>,
        /// The clips it owes and has not served yet, by name with their cost.
        owed: std::collections::HashMap<String, u64>,
        /// What is left of its budget.
        cost_left: u64,
        /// Whether it has taken a clip it does not owe.
        took_unowed: bool,
    }

    /// A pending clip that a turn's cutoff covered.
    struct AuditClip {
        arrival: Instant,
        /// The first turn whose cutoff covered it.
        first_turn: usize,
        /// The turn by which it must be served, set when its first turn
        /// leaves it over: the first turn's index plus the number of clips,
        /// itself included, that arrived no later than it and were still
        /// pending then (point 5 of the module docs).
        serve_by: Option<usize>,
    }

    /// Checks each extract against the lane rules (point 5 of the module
    /// docs) and the per-batch caps while a randomized load runs.
    #[derive(Default)]
    struct LaneAudit {
        served: Vec<String>,
        last_alone: Option<bool>,
        turn: Option<AuditTurn>,
        turns_begun: usize,
        /// Turns that began with clips left over by an earlier turn pending.
        turns_with_left_over: usize,
        /// Pending clips that a turn's cutoff covered, by name.
        clips: std::collections::HashMap<String, AuditClip>,
        /// Pending flagged item -> (flagged items ahead of it on arrival,
        /// clip turns begun since it arrived).
        flagged_waits: std::collections::HashMap<String, (usize, usize)>,
    }

    impl LaneAudit {
        /// Takes one batch of at most `max_items` and returns its names.
        fn extract(&mut self, lanes: &mut Lanes, max_items: usize) -> Vec<String> {
            // Every stub carries the hard audio cap.
            let max_cost = lanes
                .inner
                .config
                .max_batch_cost
                .min(crate::audio_prep::MAX_AUDIO_BATCH_DURATION_MS);
            let max_requests = lanes.inner.config.max_batch_requests;
            let pending: Vec<(String, bool, u64, Instant)> = lanes
                .inner
                .pending
                .iter()
                .map(|r| {
                    (
                        r.metadata.clone(),
                        r.item.alone,
                        r.item.cost,
                        r.arrival_time,
                    )
                })
                .collect();
            let batch = lanes.take_up_to(max_items);
            assert!(batch.size() > 0 && batch.size() <= max_items.min(max_requests));
            assert!(
                batch.total_cost <= max_cost || batch.size() == 1,
                "a batch of {} items costs {} ms, over the {max_cost} ms cap",
                batch.size(),
                batch.total_cost
            );
            let alone = batch.items[0].alone;
            assert!(
                batch.items.iter().all(|item| item.alone == alone),
                "lanes never share a batch"
            );
            let alone_pending = pending.iter().any(|(_, a, _, _)| *a);
            let clips_pending = pending.iter().any(|(_, a, _, _)| !*a);

            // A turn is over once none of its clips fits what is left of its
            // budget, or, before it has taken a clip it does not owe, once
            // none is pending.
            if let Some(turn) = &self.turn {
                let goes_on = pending.iter().any(|(name, _, cost, _)| {
                    turn.clips.contains_key(name) && (!turn.took_unowed || *cost <= turn.cost_left)
                });
                if !goes_on {
                    self.end_turn(&pending);
                }
            }
            if alone_pending && clips_pending {
                // The lanes take turns, starting with the lane of the oldest request.
                let expected = self.turn.is_none()
                    && self.last_alone.map_or_else(
                        || pending.iter().min_by_key(|r| r.3).unwrap().1,
                        |last| !last,
                    );
                assert_eq!(alone, expected, "lane of the next batch");
            }

            if alone {
                assert_eq!(batch.size(), 1, "a flagged item runs alone");
                let oldest = pending.iter().filter(|r| r.1).min_by_key(|r| r.3).unwrap();
                assert_eq!(
                    batch.metadata[0], oldest.0,
                    "flagged items run in arrival order"
                );
                let (ahead, clip_turns) = self
                    .flagged_waits
                    .remove(&batch.metadata[0])
                    .expect("flagged item was pending");
                assert!(
                    clip_turns <= ahead + 1,
                    "{clip_turns} clip turns with {ahead} ahead"
                );
                assert!(self.turn.is_none(), "a flagged item runs between turns");
            } else {
                if self.turn.is_none() && (alone_pending || self.last_alone == Some(true)) {
                    self.begin_turn(&pending, max_cost);
                }
                if let Some(turn) = &mut self.turn {
                    Self::check_turn_batch(turn, &batch, max_cost, max_items.min(max_requests));
                }
            }
            for name in &batch.metadata {
                self.clips.remove(name);
            }
            self.last_alone = Some(alone);
            self.served.extend(batch.metadata.iter().cloned());
            batch.metadata
        }

        /// Checks a batch of the turn under way and books it to the turn.
        fn check_turn_batch(
            turn: &mut AuditTurn,
            batch: &FormattedBatch<LaneItem, String>,
            max_cost: u64,
            max_take: usize,
        ) {
            for name in &batch.metadata {
                assert!(
                    turn.clips.contains_key(name),
                    "{name}: a turn serves only the clips pending when it began"
                );
            }
            let costs: Vec<u64> = batch.items.iter().map(|item| item.cost).collect();
            assert!(
                costs.windows(2).all(|pair| pair[0] <= pair[1]),
                "a turn's batch is in cost order: {costs:?}"
            );
            // Clips the turn does not owe go cheapest first.
            let unowed = |name: &String| !turn.owed.contains_key(name);
            let dearest_taken = batch
                .metadata
                .iter()
                .filter(|n| unowed(n))
                .map(|n| turn.clips[n])
                .max();
            let cheapest_left = turn
                .clips
                .iter()
                .filter(|(name, _)| unowed(name) && !batch.metadata.contains(name))
                .map(|(_, cost)| *cost)
                .min();
            if let (Some(taken), Some(left)) = (dearest_taken, cheapest_left) {
                assert!(
                    taken <= left,
                    "took a {taken} ms clip before a {left} ms one"
                );
            }
            // Within what is left of the budget, except for a single clip the
            // turn owes or, before it has taken a clip it does not owe, the
            // cheapest such clip.
            let owed_taken = batch
                .metadata
                .iter()
                .filter(|n| turn.owed.contains_key(*n))
                .count();
            assert!(
                batch.total_cost <= turn.cost_left
                    || (batch.size() == 1 && (owed_taken == 1 || !turn.took_unowed)),
                "turn batch of {} ms with {} ms left",
                batch.total_cost,
                turn.cost_left
            );
            let room = max_cost
                .min(turn.cost_left)
                .saturating_sub(batch.total_cost);
            for name in &batch.metadata {
                turn.clips.remove(name);
                turn.owed.remove(name);
            }
            turn.took_unowed |= owed_taken < batch.size();
            turn.cost_left = turn.cost_left.saturating_sub(batch.total_cost);
            let owed_left: u64 = turn.owed.values().sum();
            assert!(
                owed_left <= turn.cost_left,
                "the turn keeps room for the {owed_left} ms it owes"
            );
            // A batch stops short of its request limit only where no clip the
            // turn could still take fits.
            if batch.size() < max_take {
                for (name, cost) in &turn.clips {
                    let fits = *cost <= room
                        && (turn.owed.contains_key(name) || cost + owed_left <= turn.cost_left);
                    assert!(
                        !fits,
                        "a batch of {} stopped with {name} ({cost} ms) still fitting",
                        batch.size()
                    );
                }
            }
        }

        /// A turn begins: it serves the clips pending now and owes those that
        /// earlier turns left over: the oldest (and any that arrived at the
        /// same instant), then oldest first as many as fit its cost cap
        /// together. Its budget is the cost cap plus the oldest ones' cost.
        fn begin_turn(&mut self, pending: &[(String, bool, u64, Instant)], max_cost: u64) {
            self.turns_begun += 1;
            let index = self.turns_begun;
            for (name, clip) in &self.clips {
                if let Some(serve_by) = clip.serve_by {
                    assert!(
                        index <= serve_by,
                        "{name} is still pending at turn {index}, past turn {serve_by}"
                    );
                }
            }
            if !self.clips.is_empty() {
                self.turns_with_left_over += 1;
            }
            let mut left_over: Vec<(Instant, &String, u64)> = pending
                .iter()
                .filter(|r| !r.1 && self.clips.contains_key(&r.0))
                .map(|r| (r.3, &r.0, r.2))
                .collect();
            left_over.sort();
            let mut owed = std::collections::HashMap::new();
            let (mut owed_cost, mut extra) = (0, 0);
            for group in left_over.chunk_by(|a, b| a.0 == b.0) {
                let cost: u64 = group.iter().map(|&(_, _, cost)| cost).sum();
                if owed.is_empty() {
                    extra = cost;
                } else if owed_cost + cost > max_cost {
                    break;
                }
                owed_cost += cost;
                owed.extend(group.iter().map(|&(_, name, cost)| (name.clone(), cost)));
            }
            let mut clips = std::collections::HashMap::new();
            for (name, _, cost, arrival) in pending.iter().filter(|r| !r.1) {
                clips.insert(name.clone(), *cost);
                self.clips.entry(name.clone()).or_insert(AuditClip {
                    arrival: *arrival,
                    first_turn: index,
                    serve_by: None,
                });
            }
            for wait in self.flagged_waits.values_mut() {
                wait.1 += 1;
            }
            self.turn = Some(AuditTurn {
                index,
                clips,
                owed,
                cost_left: max_cost + extra,
                took_unowed: false,
            });
        }

        /// The turn under way is over: it served everything it owed. Set the
        /// bound of each clip it was the first turn of and left over.
        fn end_turn(&mut self, pending: &[(String, bool, u64, Instant)]) {
            let turn = self.turn.take().expect("a turn is under way");
            assert!(
                turn.owed.is_empty(),
                "turn {} ended owing {:?}",
                turn.index,
                turn.owed.keys().collect::<Vec<_>>()
            );
            let arrivals: Vec<Instant> = pending.iter().filter(|r| !r.1).map(|r| r.3).collect();
            for clip in self.clips.values_mut() {
                if clip.first_turn == turn.index {
                    let ahead = arrivals.iter().filter(|a| **a <= clip.arrival).count();
                    clip.serve_by = Some(turn.index + ahead);
                }
            }
        }
    }

    #[test]
    fn lanes_never_mix_and_take_turns_at_any_cost_cap() {
        // The default adaptive cost range (4096-65536 ms) and its 16384 ms
        // start, for backends that report no budget; a speech model's floor
        // (180000 ms) and its 720000 ms start, which is also the hard audio
        // cap.
        for max_batch_cost in [4_096, 16_384, 65_536, 180_000, 720_000] {
            let mut rng = TestRng(max_batch_cost);
            let mut lanes = Lanes::new(max_batch_cost);
            let mut audit = LaneAudit::default();
            let mut submitted = Vec::new();
            let mut now_ms = 0;

            for index in 0..300 {
                now_ms += rng.range(1, 2_000);
                if rng.chance(60) {
                    let name = format!("item-{index}");
                    let item = if rng.chance(30) {
                        let ahead = audit.flagged_waits.len();
                        audit.flagged_waits.insert(name.clone(), (ahead, 0));
                        long_form_ms(rng.range(30_001, 300_000))
                    } else {
                        clip_ms(rng.range(1_000, 30_000))
                    };
                    lanes.arrive(item, &name, now_ms);
                    submitted.push(name);
                }
                if !lanes.inner.pending.is_empty() && rng.chance(50) {
                    let max_items = rng.pick(&[usize::MAX, 1, 3]);
                    audit.extract(&mut lanes, max_items);
                }
            }
            while !lanes.inner.pending.is_empty() {
                let max_items = rng.pick(&[usize::MAX, 1, 3]);
                audit.extract(&mut lanes, max_items);
            }

            audit.served.sort();
            submitted.sort();
            assert_eq!(audit.served, submitted, "cap {max_batch_cost}");
            assert!(audit.flagged_waits.is_empty());
            assert_eq!(lanes.inner.total_cost, 0);
        }
    }

    /// Queues `item` one ms after the previous arrival, under a fresh name
    /// that starts with `kind`.
    fn arrive_next(
        lanes: &mut Lanes,
        audit: &mut LaneAudit,
        item: LaneItem,
        kind: &str,
        now_ms: &mut u64,
    ) {
        *now_ms += 1;
        let name = format!("{kind}-{now_ms}");
        if item.alone {
            let ahead = audit.flagged_waits.len();
            audit.flagged_waits.insert(name.clone(), (ahead, 0));
        }
        lanes.arrive(item, &name, *now_ms);
    }

    #[test]
    fn every_clip_is_served_within_its_turn_bound_under_replenishment() {
        // Long recordings stay pending throughout (their clients resubmit at
        // once), so clip turns never stop. Clip clients resubmit at once too,
        // and one-off bursts add more clips. Clips cost up to 30 s, above the
        // cap at the lower caps; drain budgets cut some batches short, and the
        // cost cap moves between batches as the adaptive controller moves it.
        // The audit checks every batch against the caps and the lane rules,
        // and every clip against its bound: a clip its first turn leaves over
        // is served by the turn numbered that turn plus the clips, itself
        // included, that arrived no later than it and were still pending then.
        let caps = [4_096, 16_384, 65_536, 180_000, 720_000];
        let mut turns_with_left_over = 0;
        for seed in 0..24 {
            let mut rng = TestRng(1_000 + seed);
            let mut lanes = Lanes::new(rng.pick(&caps));
            let mut audit = LaneAudit::default();
            let mut now_ms = 0;
            for _ in 0..rng.range(1, 3) {
                let item = long_form_ms(rng.range(30_001, 300_000));
                arrive_next(&mut lanes, &mut audit, item, "recording", &mut now_ms);
            }
            for _ in 0..rng.range(4, 48) {
                let item = clip_ms(rng.range(1_000, 30_000));
                arrive_next(&mut lanes, &mut audit, item, "client", &mut now_ms);
            }

            for _ in 0..400 {
                if rng.chance(10) {
                    lanes.inner.config.max_batch_cost = rng.pick(&caps);
                }
                if rng.chance(10) {
                    for _ in 0..rng.range(1, 8) {
                        let item = clip_ms(rng.range(1_000, 30_000));
                        arrive_next(&mut lanes, &mut audit, item, "burst", &mut now_ms);
                    }
                }
                let max_items = rng.pick(&[usize::MAX, usize::MAX, 1, 3, 5]);
                for name in audit.extract(&mut lanes, max_items) {
                    if name.starts_with("recording") {
                        let item = long_form_ms(rng.range(30_001, 300_000));
                        arrive_next(&mut lanes, &mut audit, item, "recording", &mut now_ms);
                    } else if name.starts_with("client") {
                        let item = clip_ms(rng.range(1_000, 30_000));
                        arrive_next(&mut lanes, &mut audit, item, "client", &mut now_ms);
                    }
                }
            }
            assert!(
                audit.turns_begun >= 70,
                "seed {seed}: clip turns keep coming"
            );
            turns_with_left_over += audit.turns_with_left_over;
        }
        assert!(
            turns_with_left_over >= 1_500,
            "the load leaves clips over: {turns_with_left_over} turns began with some"
        );
    }
}
