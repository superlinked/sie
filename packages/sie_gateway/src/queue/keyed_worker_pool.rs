//! Bounded worker pool keyed by request id.
//!
//! The gateway inbox is one subscription feeding one tokio task
//! (`publisher::WorkPublisher::handle_inbox`). Before this module that task
//! also *ran* every handler inline, so one slow handler — most sharply an
//! object-store delete on request completion — stalled decode for every other
//! in-flight request in the process. For streamed generation that is every
//! token chunk of every concurrent stream behind one await.
//!
//! This pool splits the *running* from the *decoding*. The decode task stays
//! single (the msgpack decode still happens exactly once, on that task) and
//! hands each decoded item to one of a fixed number of shards. A shard is a
//! tokio task draining a bounded channel strictly in order.
//!
//! # Ordering
//!
//! A key always maps to the same shard ([`shard_index`] is a pure function of
//! the key), and a shard drains its channel in FIFO order and awaits each
//! handler to completion before taking the next item. Therefore **two items
//! with the same key are handled in dispatch order**, which is the order the
//! decode task read them off the subscription. That is the property streaming
//! depends on: `StreamCollector::apply` drops a chunk whose `seq` is not
//! greater than the last applied one and fails the whole stream on a forward
//! gap, so a reordered pair of chunks would truncate or kill a live stream.
//!
//! Items with *different* keys may land on different shards and then run
//! concurrently. Cross-key ordering was never a property of the old inline
//! loop that anything depended on — every request's state is keyed by request
//! id in its own `DashMap` entry.
//!
//! # Bound
//!
//! Each shard's channel has a fixed depth, so a slow handler cannot grow an
//! unbounded queue of decoded payloads. Two policies are offered at the bound
//! and callers pick per workload:
//!
//! * [`KeyedWorkerPool::dispatch`] — **backpressure**. Awaits a free slot.
//!   Correct for work that must not be lost: at worst the decode task blocks,
//!   which is exactly what it did for *every* message before this change, so
//!   saturation degrades to the old behaviour rather than to a new failure
//!   mode. Nothing is dropped.
//! * [`KeyedWorkerPool::try_dispatch`] — **shed**. Returns `false` instead of
//!   waiting. Correct only for work that has an independent retry path, so
//!   that shedding costs latency and not correctness.

use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::observability::metrics::{self as telemetry, QueueWorkerPoolEvent};

/// Map a key onto one of `shards` ordering domains.
///
/// FNV-1a, written out rather than taken from `DefaultHasher` so the mapping
/// is deterministic across processes and toolchain versions: tests pick keys
/// that land on distinct shards by calling this directly, and an operator
/// reading a shard label in a log can reproduce the assignment.
pub(crate) fn shard_index(key: &str, shards: usize) -> usize {
    debug_assert!(shards > 0, "shard_index needs at least one shard");
    const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET_BASIS;
    for byte in key.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    (hash % shards.max(1) as u64) as usize
}

/// Work that carries its own ordering key.
///
/// The key travels with the item so a caller cannot accidentally dispatch an
/// item under the wrong key and silently break its ordering domain.
pub(crate) trait ShardKeyed {
    fn shard_key(&self) -> &str;
}

struct Shard<T> {
    sender: mpsc::Sender<T>,
    /// Edge trigger for the saturation warning. Sustained saturation would
    /// otherwise emit one warning per message; the counter carries the volume.
    saturated: AtomicBool,
}

/// A fixed set of ordered, bounded worker tasks addressed by key.
///
/// See the module docs for the ordering guarantee and the two bound policies.
/// Dropping the pool closes every shard channel, which ends the shard tasks
/// once they finish draining.
pub(crate) struct KeyedWorkerPool<T> {
    /// Bounded-cardinality metric label identifying this pool.
    name: &'static str,
    shards: Vec<Shard<T>>,
}

impl<T: Send + 'static> KeyedWorkerPool<T> {
    /// Spawn `shards` worker tasks, each draining a channel of depth `depth`.
    ///
    /// Shards are ordering domains, not threads: they are ordinary tokio tasks
    /// multiplexed onto the runtime's existing workers, so the count is chosen
    /// for concurrency width and bounded buffering, not for core count.
    pub(crate) fn new<F, Fut>(name: &'static str, shards: usize, depth: usize, handler: F) -> Self
    where
        F: Fn(T) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        assert!(shards > 0, "keyed worker pool needs at least one shard");
        assert!(depth > 0, "keyed worker pool needs a non-zero queue depth");

        let handler = Arc::new(handler);
        let shards = (0..shards)
            .map(|_| {
                let (sender, mut receiver) = mpsc::channel::<T>(depth);
                let handler = Arc::clone(&handler);
                tokio::spawn(async move {
                    // Sequential drain. This `await` is what makes same-shard
                    // (and therefore same-key) handling strictly ordered.
                    while let Some(item) = receiver.recv().await {
                        handler(item).await;
                    }
                });
                Shard {
                    sender,
                    saturated: AtomicBool::new(false),
                }
            })
            .collect::<Vec<_>>();

        info!(
            pool = name,
            shards = shards.len(),
            depth,
            "started keyed worker pool"
        );
        Self { name, shards }
    }

    /// Hand `item` to the shard owning its key, waiting for a free slot.
    ///
    /// Returns `false` only when the pool's worker task is gone, which can
    /// only happen if it panicked — the caller should treat that as fatal for
    /// its own loop rather than spin.
    pub(crate) async fn dispatch(&self, item: T) -> bool
    where
        T: ShardKeyed,
    {
        let index = shard_index(item.shard_key(), self.shards.len());
        let shard = &self.shards[index];
        let item = match shard.sender.try_send(item) {
            Ok(()) => {
                shard.saturated.store(false, Ordering::Relaxed);
                return true;
            }
            Err(mpsc::error::TrySendError::Full(item)) => item,
            Err(mpsc::error::TrySendError::Closed(_)) => {
                warn!(pool = self.name, shard = index, "worker pool shard closed");
                return false;
            }
        };

        telemetry::record_queue_worker_pool_event(self.name, QueueWorkerPoolEvent::Saturated);
        if !shard.saturated.swap(true, Ordering::Relaxed) {
            warn!(
                pool = self.name,
                shard = index,
                depth = shard.sender.max_capacity(),
                "worker pool shard saturated — applying backpressure to the caller"
            );
        }
        if shard.sender.send(item).await.is_err() {
            warn!(pool = self.name, shard = index, "worker pool shard closed");
            return false;
        }
        true
    }

    /// Hand `item` to the shard owning its key without waiting.
    ///
    /// Returns `false` when the shard is saturated (or closed) and the item
    /// was **not** accepted. Only use this where the caller has an independent
    /// path that will redo the work.
    pub(crate) fn try_dispatch(&self, item: T) -> bool
    where
        T: ShardKeyed,
    {
        let index = shard_index(item.shard_key(), self.shards.len());
        let shard = &self.shards[index];
        match shard.sender.try_send(item) {
            Ok(()) => {
                shard.saturated.store(false, Ordering::Relaxed);
                true
            }
            Err(mpsc::error::TrySendError::Full(_)) => {
                telemetry::record_queue_worker_pool_event(self.name, QueueWorkerPoolEvent::Shed);
                if !shard.saturated.swap(true, Ordering::Relaxed) {
                    warn!(
                        pool = self.name,
                        shard = index,
                        depth = shard.sender.max_capacity(),
                        "worker pool shard saturated — shedding to the caller's retry path"
                    );
                }
                false
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                debug!(pool = self.name, shard = index, "worker pool shard closed");
                false
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::sync::Mutex;
    use std::time::Duration;

    /// Every bound test uses one key so it addresses exactly one shard.
    const KEY: &str = "k";

    /// A keyed chunk: `key` is the ordering domain, `seq` the position in it.
    #[derive(Debug)]
    struct Keyed {
        key: String,
        seq: u32,
    }

    impl ShardKeyed for Keyed {
        fn shard_key(&self) -> &str {
            &self.key
        }
    }

    impl ShardKeyed for String {
        fn shard_key(&self) -> &str {
            self
        }
    }

    /// Two keys that provably land on different shards, so the concurrency
    /// tests below assert real concurrency instead of hash luck.
    fn keys_on_distinct_shards(shards: usize) -> (String, String) {
        let first = "req-0".to_string();
        let first_shard = shard_index(&first, shards);
        for candidate in 1..1_000 {
            let key = format!("req-{candidate}");
            if shard_index(&key, shards) != first_shard {
                return (first, key);
            }
        }
        panic!("no two keys landed on distinct shards");
    }

    #[test]
    fn shard_index_is_stable_and_in_range() {
        for shards in [1usize, 2, 4, 8, 13] {
            for candidate in 0..256 {
                let key = format!("request-{candidate}");
                let index = shard_index(&key, shards);
                assert!(index < shards, "shard index must be in range");
                assert_eq!(
                    index,
                    shard_index(&key, shards),
                    "shard index must be a pure function of the key"
                );
            }
        }
    }

    /// The load-bearing property: interleave many sequences across several
    /// keys and every key's handler must observe strictly increasing `seq`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn preserves_per_key_order_under_concurrency() {
        const KEYS: usize = 16;
        const PER_KEY: u32 = 64;

        let seen: Arc<Mutex<BTreeMap<String, Vec<u32>>>> = Arc::new(Mutex::new(BTreeMap::new()));
        let recorder = Arc::clone(&seen);
        let pool = KeyedWorkerPool::new("test-order", 8, 4, move |item: Keyed| {
            let recorder = Arc::clone(&recorder);
            async move {
                // Yield inside the handler so a broken implementation that
                // spawned per item (rather than draining in order) would
                // interleave and be caught.
                tokio::task::yield_now().await;
                recorder
                    .lock()
                    .expect("recorder")
                    .entry(item.key)
                    .or_default()
                    .push(item.seq);
            }
        });

        // Round-robin across keys so each shard's channel sees an interleaved
        // stream, exactly like the real inbox under concurrent streams.
        for seq in 0..PER_KEY {
            for key_index in 0..KEYS {
                let key = format!("req-{key_index}");
                assert!(pool.dispatch(Keyed { key, seq }).await, "dispatch accepted");
            }
        }
        drop(pool);

        // Drain: wait until every key has all its items.
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            let complete = {
                let guard = seen.lock().expect("recorder");
                guard.len() == KEYS && guard.values().all(|seqs| seqs.len() == PER_KEY as usize)
            };
            if complete {
                break;
            }
            assert!(std::time::Instant::now() < deadline, "pool did not drain");
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        let guard = seen.lock().expect("recorder");
        for (key, seqs) in guard.iter() {
            assert_eq!(
                *seqs,
                (0..PER_KEY).collect::<Vec<_>>(),
                "key {key} must observe strictly increasing seq in dispatch order"
            );
        }
    }

    /// If the pool ever serialized everything again this deadlocks: the first
    /// key's handler only returns once the second key's handler has run.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn distinct_keys_proceed_concurrently() {
        let (blocking_key, other_key) = keys_on_distinct_shards(8);
        let unblock = Arc::new(tokio::sync::Notify::new());
        let completed = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let handler_unblock = Arc::clone(&unblock);
        let handler_completed = Arc::clone(&completed);
        let blocking = blocking_key.clone();

        let pool = KeyedWorkerPool::new("test-concurrency", 8, 4, move |key: String| {
            let unblock = Arc::clone(&handler_unblock);
            let completed = Arc::clone(&handler_completed);
            let blocking = blocking.clone();
            async move {
                if key == blocking {
                    // Held until the *other* key's item has been handled. If
                    // the pool serialized every key onto one worker, the
                    // other key's item would sit behind this await forever.
                    unblock.notified().await;
                } else {
                    unblock.notify_one();
                }
                completed.fetch_add(1, Ordering::SeqCst);
            }
        });

        assert!(pool.dispatch(blocking_key.clone()).await);
        assert!(pool.dispatch(other_key.clone()).await);

        tokio::time::timeout(Duration::from_secs(5), async {
            while completed.load(Ordering::SeqCst) < 2 {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("distinct keys must not serialize behind each other");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dispatch_applies_backpressure_when_saturated() {
        let release = Arc::new(tokio::sync::Notify::new());
        let handler_release = Arc::clone(&release);
        let pool = KeyedWorkerPool::new("test-backpressure", 1, 1, move |_: Keyed| {
            let release = Arc::clone(&handler_release);
            async move {
                release.notified().await;
            }
        });

        // One item is taken by the worker (which then blocks) and one fills
        // the depth-1 channel. The pool is now saturated.
        assert!(
            pool.dispatch(Keyed {
                key: KEY.into(),
                seq: 0
            })
            .await
        );
        assert!(
            pool.dispatch(Keyed {
                key: KEY.into(),
                seq: 1
            })
            .await
        );
        tokio::time::sleep(Duration::from_millis(50)).await;

        // The third dispatch must wait rather than drop or grow the queue.
        let blocked = tokio::time::timeout(
            Duration::from_millis(200),
            pool.dispatch(Keyed {
                key: KEY.into(),
                seq: 2,
            }),
        )
        .await;
        assert!(
            blocked.is_err(),
            "dispatch must block at the bound instead of over-filling"
        );

        // Releasing the handler lets the queue drain and the dispatch land.
        release.notify_waiters();
        let drained = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                release.notify_waiters();
                if pool
                    .dispatch(Keyed {
                        key: KEY.into(),
                        seq: 3,
                    })
                    .await
                {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await;
        assert!(drained.is_ok(), "dispatch must complete once slots free up");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn try_dispatch_sheds_when_saturated() {
        let release = Arc::new(tokio::sync::Notify::new());
        let handler_release = Arc::clone(&release);
        let pool = KeyedWorkerPool::new("test-shed", 1, 1, move |_: Keyed| {
            let release = Arc::clone(&handler_release);
            async move {
                release.notified().await;
            }
        });

        assert!(pool.try_dispatch(Keyed {
            key: KEY.into(),
            seq: 0
        }));
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            pool.try_dispatch(Keyed {
                key: KEY.into(),
                seq: 1
            }),
            "the depth-1 channel accepts one"
        );
        assert!(
            !pool.try_dispatch(Keyed {
                key: KEY.into(),
                seq: 2
            }),
            "a saturated shard must shed rather than block"
        );

        release.notify_waiters();
        let accepted = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                release.notify_waiters();
                if pool.try_dispatch(Keyed {
                    key: KEY.into(),
                    seq: 3,
                }) {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await;
        assert!(accepted.is_ok(), "shedding must be transient, not sticky");
    }
}
