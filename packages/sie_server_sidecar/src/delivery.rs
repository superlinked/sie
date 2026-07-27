//! Delivery abstraction: where a `WorkItem` came from and where its
//! settlement (result publish / ACK / redelivery request) must go.
//!
//! Added for the local-ingest mode: in a broker-less sidecar lane there
//! is no NATS, so the dispatcher cannot settle
//! work by ACK/NAK-ing a JetStream [`Message`] or publishing to a reply
//! subject. Instead of forking the dispatch pipeline, the pipeline becomes
//! generic over this small enum:
//!
//! * [`Delivery::Nats`] — the existing JetStream message. ACK/NAK/progress
//!   map 1:1 onto the underlying [`Message`]; results still publish to
//!   `WorkItem.reply_subject` via [`crate::publisher::WorkPublisher`].
//!   **Zero behaviour change** for the NATS path.
//! * [`Delivery::Local`] — one slot of an in-flight local-ingest
//!   `publish_work` call (`crate::local_ingest`). Results are sent back to
//!   the ingest connection as [`LocalDeliveryEvent::Result`]; a NAK becomes
//!   [`LocalDeliveryEvent::Retry`] (there is no broker to redeliver, so the
//!   ingest layer re-dispatches with a bounded attempt budget — the same
//!   proportional stand-in the reference Python lane uses); ACK and progress
//!   are no-ops because the terminal `Result` event *is* the settlement.

use std::time::Duration;

use async_nats::jetstream::Message;
use tokio::sync::mpsc;
use tokio::sync::OwnedSemaphorePermit;

use crate::work_types::WorkResult;

/// Event emitted by the dispatcher for one local-ingest slot.
#[derive(Debug)]
pub enum LocalDeliveryEvent {
    /// Terminal settlement: a `WorkResult` (success or typed error) for
    /// `slot`. Exactly one per slot reaches the ingest response. Boxed:
    /// `WorkResult` dwarfs `Retry`, and the event only crosses a channel
    /// once per item, so the indirection is free.
    Result {
        slot: usize,
        result: Box<WorkResult>,
    },
    /// The worker requested redelivery (the NATS NAK analogue). The
    /// ingest layer re-dispatches the original `WorkItem` after
    /// `delay_ms`, or synthesises a typed error once the attempt budget
    /// is exhausted. `attempt` is the attempt that just NAKed (0-based).
    Retry {
        slot: usize,
        attempt: u32,
        delay_ms: u64,
    },
}

/// Settlement handle for one item of a local-ingest `publish_work` call.
#[derive(Debug, Clone)]
pub struct LocalDelivery {
    slot: usize,
    attempt: u32,
    tx: mpsc::UnboundedSender<LocalDeliveryEvent>,
}

impl LocalDelivery {
    pub fn new(slot: usize, attempt: u32, tx: mpsc::UnboundedSender<LocalDeliveryEvent>) -> Self {
        Self { slot, attempt, tx }
    }

    pub fn slot(&self) -> usize {
        self.slot
    }

    pub fn attempt(&self) -> u32 {
        self.attempt
    }

    /// Send the terminal result. `false` means the ingest caller is gone
    /// (connection closed / call timed out) — there is nowhere left to
    /// deliver, so callers just log and move on.
    pub fn send_result(&self, result: WorkResult) -> bool {
        self.tx
            .send(LocalDeliveryEvent::Result {
                slot: self.slot,
                result: Box::new(result),
            })
            .is_ok()
    }

    fn send_retry(&self, delay_ms: u64) -> bool {
        self.tx
            .send(LocalDeliveryEvent::Retry {
                slot: self.slot,
                attempt: self.attempt,
                delay_ms,
            })
            .is_ok()
    }
}

/// The dispatcher-facing settlement token. Not `Clone` on the NATS arm:
/// the JetStream [`Message`] owns the ACK/NAK token, which must not be
/// duplicated (same single-ownership rule `SchedulerMeta` documents).
///
/// Variant sizes are lopsided (`Message` ≈ 400 B vs 24 B), but this enum
/// occupies exactly the slots that held a bare `Message` before the
/// local-ingest refactor — boxing it would add a heap hop to every NATS
/// settlement to save memory the pipeline was already spending. Scoped
/// allow, same policy as `publisher::ShapeOutcome`.
#[allow(clippy::large_enum_variant)]
pub enum Delivery {
    Nats(
        Message,
        /// Pull-loop admission permit: intake capacity stays occupied until
        /// this delivery settles (ACK / NAK / drop) — exactly the
        /// queue-admission bound. Local-ingest deliveries carry no permit.
        Option<OwnedSemaphorePermit>,
    ),
    Local(LocalDelivery),
}

impl Delivery {
    /// True when this delivery arrived worker-directly rather than via the
    /// pool subject: a NATS message on a worker-specific direct-dispatch
    /// subject, or any local-ingest delivery (the caller addressed THIS
    /// worker's socket). Feeds `WorkResult.worker_direct` /
    /// `SchedulerMeta.worker_direct` so the gateway's direct-fallback
    /// cancellation bookkeeping (main's NAK/stale-republish hardening) sees
    /// the same signal on every path.
    pub fn worker_direct(&self) -> bool {
        match self {
            Self::Nats(msg, _) => crate::subject::is_worker_direct_work_subject(&msg.subject),
            Self::Local(_) => true,
        }
    }

    /// ACK — "settled, never redeliver". Local deliveries settle via
    /// their terminal [`LocalDeliveryEvent::Result`], so this is a no-op.
    pub async fn ack(&self) -> Result<(), String> {
        match self {
            Self::Nats(msg, _) => msg.ack().await.map_err(|e| e.to_string()),
            Self::Local(_) => Ok(()),
        }
    }

    /// NAK — "not settled, redeliver after `delay_ms`". Local deliveries
    /// route this to the ingest layer's bounded re-dispatch.
    pub async fn nak(&self, delay_ms: u64) -> Result<(), String> {
        match self {
            Self::Nats(msg, _) => msg
                .ack_with(async_nats::jetstream::AckKind::Nak(Some(
                    Duration::from_millis(delay_ms),
                )))
                .await
                .map_err(|e| e.to_string()),
            Self::Local(local) => {
                if local.send_retry(delay_ms) {
                    Ok(())
                } else {
                    Err("local ingest caller gone — retry event dropped".to_string())
                }
            }
        }
    }

    /// Progress ACK — "still working, reset the redelivery clock". The
    /// local ingest caller holds one synchronous call open with no
    /// ack-wait clock, so this is a no-op there.
    pub async fn progress(&self) -> Result<(), String> {
        match self {
            Self::Nats(msg, _) => msg
                .ack_with(async_nats::jetstream::AckKind::Progress)
                .await
                .map_err(|e| e.to_string()),
            Self::Local(_) => Ok(()),
        }
    }

    /// Log-friendly origin reference (NATS subject / local slot).
    pub fn log_ref(&self) -> String {
        match self {
            Self::Nats(msg, _) => msg.subject.to_string(),
            Self::Local(local) => format!("local[slot={},attempt={}]", local.slot, local.attempt),
        }
    }
}

impl std::fmt::Debug for Delivery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `Message` intentionally has no useful Debug; print the origin ref.
        f.debug_tuple("Delivery").field(&self.log_ref()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(slot: usize) -> WorkResult {
        WorkResult {
            work_item_id: format!("r.{slot}"),
            request_id: "r".into(),
            item_index: slot as u32,
            success: true,
            result_msgpack: vec![1, 2, 3],
            error: None,
            error_code: None,
            worker_direct: true,
            inference_ms: None,
            queue_ms: None,
            processing_ms: None,
            worker_id: Some("w".into()),
            tokenization_ms: None,
            postprocessing_ms: None,
            payload_fetch_ms: None,
            units: None,
            executed_bundle_config_hash: None,
        }
    }

    #[tokio::test]
    async fn local_ack_and_progress_are_noops() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let d = Delivery::Local(LocalDelivery::new(0, 0, tx));
        d.ack().await.expect("local ack is Ok");
        d.progress().await.expect("local progress is Ok");
        // Neither emits an event.
        assert!(rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn local_nak_emits_retry_with_attempt_and_delay() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let d = Delivery::Local(LocalDelivery::new(3, 1, tx));
        d.nak(250).await.expect("retry event sent");
        match rx.try_recv().expect("event present") {
            LocalDeliveryEvent::Retry {
                slot,
                attempt,
                delay_ms,
            } => {
                assert_eq!(slot, 3);
                assert_eq!(attempt, 1);
                assert_eq!(delay_ms, 250);
            }
            other => panic!("expected Retry, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn local_nak_after_caller_gone_is_error() {
        let (tx, rx) = mpsc::unbounded_channel();
        drop(rx);
        let d = Delivery::Local(LocalDelivery::new(0, 0, tx));
        assert!(d.nak(100).await.is_err());
    }

    #[tokio::test]
    async fn local_send_result_round_trips() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let local = LocalDelivery::new(2, 0, tx);
        assert!(local.send_result(result(2)));
        match rx.try_recv().expect("event present") {
            LocalDeliveryEvent::Result { slot, result } => {
                assert_eq!(slot, 2);
                assert!(result.success);
                assert_eq!(result.work_item_id, "r.2");
            }
            other @ LocalDeliveryEvent::Retry { .. } => panic!("expected Result, got {other:?}"),
        }
    }

    #[test]
    fn log_ref_identifies_local_slot() {
        let (tx, _rx) = mpsc::unbounded_channel();
        let d = Delivery::Local(LocalDelivery::new(7, 2, tx));
        assert_eq!(d.log_ref(), "local[slot=7,attempt=2]");
    }
}
