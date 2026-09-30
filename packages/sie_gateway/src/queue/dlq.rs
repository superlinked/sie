use std::time::Duration;

use async_nats::jetstream;
use futures_util::StreamExt;
use tracing::{debug, error, info, warn};

use super::stream_durability;
use crate::nats::manager::server_originated;
use crate::observability::metrics::{self as telemetry, QueueEvent, QueueEventOutcome};

const DLQ_STREAM_NAME: &str = "DEAD_LETTERS";
const DLQ_SUBJECT: &str = "sie.dlq.>";
const DLQ_RETENTION_SECS: u64 = 86400; // 24 hours
const ADVISORY_SUBJECT: &str = "$JS.EVENT.ADVISORY.CONSUMER.MAX_DELIVERIES.>";

pub struct DlqListener;

impl DlqListener {
    /// Create-or-reconcile the dead-letter stream.
    ///
    /// Split out of [`Self::start`] and parameterized by name/subject purely
    /// so the live test can drive this logic against a uniquely-named stream.
    /// `DEAD_LETTERS` is a single fixed-name stream shared by every gateway
    /// replica on a broker, so a test that operated on the real name would be
    /// destructive well beyond its own scope — and deleting it to set up a
    /// test would be precisely the delete-and-recreate this function exists to
    /// avoid. Production always passes [`DLQ_STREAM_NAME`]/[`DLQ_SUBJECT`].
    ///
    /// Storage tracks the work-queue setting (`SIE_STREAM_STORAGE`, default
    /// `Memory`): a memory-backed DLQ is erased by the same broker restart
    /// that erases the work it was recording, which is the worst possible
    /// time to lose the forensic record. Enabling file storage for the work
    /// queue therefore enables it here too, from the same knob.
    async fn ensure_dlq_stream(
        jetstream: &jetstream::Context,
        stream_name: &str,
        subject: &str,
        storage: jetstream::stream::StorageType,
        num_replicas: usize,
    ) -> Result<(), String> {
        let num_replicas = num_replicas.max(1);
        let mut stream = jetstream
            .get_or_create_stream(jetstream::stream::Config {
                name: stream_name.to_string(),
                subjects: vec![subject.to_string()],
                retention: jetstream::stream::RetentionPolicy::Limits,
                storage,
                num_replicas,
                max_age: Duration::from_secs(DLQ_RETENTION_SECS),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("create DLQ stream: {}", e))?;

        // `get_or_create_stream` silently ignores the requested config when
        // the stream already exists, so a DLQ created by an earlier release
        // (or before the durability knobs existed) would keep its old shape
        // with no signal at all. Reconcile it exactly like the work streams:
        // replicas repaired, storage reported but never repaired.
        //
        // Emphatically NOT delete-and-recreate. The DLQ is the forensic
        // record of work that has already been lost once; destroying it to
        // change its storage type would be the same class of failure this
        // whole knob exists to fix.
        stream_durability::reconcile_num_replicas(
            jetstream,
            &mut stream,
            stream_name,
            num_replicas,
        )
        .await?;
        stream_durability::warn_on_storage_mismatch(&stream, stream_name, storage);

        info!(
            stream = stream_name,
            retention_hours = DLQ_RETENTION_SECS / 3600,
            storage = ?storage,
            num_replicas,
            "dead letter queue stream ready"
        );
        Ok(())
    }

    /// Start listening for NATS JetStream advisory events and routing
    /// max-delivery messages to a dead letter stream.
    pub async fn start(
        jetstream: jetstream::Context,
        client: async_nats::Client,
        storage: jetstream::stream::StorageType,
        num_replicas: usize,
    ) -> Result<(), String> {
        Self::ensure_dlq_stream(
            &jetstream,
            DLQ_STREAM_NAME,
            DLQ_SUBJECT,
            storage,
            num_replicas,
        )
        .await?;

        // Subscribe every gateway replica to max-delivery advisories. The
        // publish into DEAD_LETTERS is stamped with a deterministic message id
        // below, so JetStream dedupes the fan-out while preserving HA: if one
        // replica sees the advisory but fails before publishing, another
        // replica can still persist it.
        let subscriber = client
            .subscribe(ADVISORY_SUBJECT.to_string())
            .await
            .map_err(|e| format!("subscribe to advisory: {}", e))?;

        let js = jetstream.clone();
        tokio::spawn(async move {
            Self::handle_advisories(subscriber, js).await;
        });

        info!(subject = ADVISORY_SUBJECT, "DLQ advisory listener started");

        Ok(())
    }

    async fn handle_advisories(
        mut subscriber: async_nats::Subscriber,
        jetstream: jetstream::Context,
    ) {
        while let Some(msg) = subscriber.next().await {
            let subject = msg.subject.as_str();
            if let Some(reason) = server_originated(&msg) {
                warn!(
                    subject = %subject,
                    reason,
                    "dropping max-deliveries advisory that the server did not emit itself"
                );
                continue;
            }
            let payload = msg.payload.to_vec();

            // Parse the advisory to extract stream/consumer info
            let advisory: serde_json::Value = match serde_json::from_slice(&payload) {
                Ok(v) => v,
                Err(e) => {
                    warn!(
                        subject = %subject,
                        error = %e,
                        "failed to parse advisory JSON"
                    );
                    continue;
                }
            };

            let stream_name = advisory
                .get("stream")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let consumer_name = advisory
                .get("consumer")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let stream_seq = advisory
                .get("stream_seq")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let deliveries = advisory
                .get("deliveries")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);

            warn!(
                stream = %stream_name,
                consumer = %consumer_name,
                stream_seq = stream_seq,
                deliveries = deliveries,
                "message exceeded max deliveries"
            );

            // Extract the model token from the advisory's original subject.
            //
            // The publisher constructs work subjects as
            // `sie.work.{pool}.{machine_profile}.{bundle}.{normalize_model_id(model)}`
            // — exactly six dot-separated tokens, with the 6th token already
            // safe to use as a single NATS token (no `.`, `*`, `>`, or whitespace).
            // We keep
            // the `/` -> `_` belt-and-suspenders replacement as a no-op for
            // correctly-normalized subjects and a fallback for any legacy
            // un-normalized messages still in-flight at the time of upgrade.
            let model_normalized = advisory
                .get("subject")
                .and_then(|v| v.as_str())
                .and_then(|s| {
                    let parts: Vec<&str> = s.split('.').collect();
                    if parts.len() >= 6 {
                        Some(parts[5].replace('/', "_"))
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| format!("{}.{}.{}", stream_name, consumer_name, stream_seq));

            // Forward the advisory payload to the DLQ stream
            let dlq_subject = format!("sie.dlq.{}", model_normalized);
            let message_id = dlq_message_id(&advisory, stream_name, consumer_name, stream_seq);

            // `jetstream.publish(...).await` returns a
            // `PublishAckFuture` once the client has queued the
            // message; the server's ack (or NAK) lands when we await
            // that future. Without the second await we'd miss
            // server-side rejections (stream doesn't exist, quota
            // exceeded, consumer backpressure) and the failure
            // counter would undercount real outages. The cost is a
            // per-message round-trip, which is acceptable here — DLQ
            // is a rare, degraded-state path, not the hot inference
            // loop.
            let publish_result: Result<jetstream::publish::PublishAck, String> = match jetstream
                .send_publish(
                    dlq_subject.clone(),
                    jetstream::message::PublishMessage::build()
                        .message_id(&message_id)
                        .payload(payload.into()),
                )
                .await
            {
                Ok(ack_future) => ack_future.await.map_err(|e| e.to_string()),
                Err(e) => Err(e.to_string()),
            };
            match publish_result {
                Ok(ack) if ack.duplicate => {
                    telemetry::record_queue_event(
                        QueueEvent::DlqForward,
                        QueueEventOutcome::Deduplicated,
                    );
                    debug!(
                        subject = %dlq_subject,
                        stream = %stream_name,
                        seq = stream_seq,
                        message_id = %message_id,
                        "duplicate DLQ advisory publish deduped by JetStream"
                    );
                }
                Ok(_) => {
                    telemetry::record_queue_event(
                        QueueEvent::DlqForward,
                        QueueEventOutcome::Success,
                    );
                    info!(
                        subject = %dlq_subject,
                        stream = %stream_name,
                        seq = stream_seq,
                        "forwarded dead letter to DLQ"
                    );
                }
                Err(e) => {
                    telemetry::record_queue_event(QueueEvent::DlqForward, QueueEventOutcome::Error);
                    error!(
                        subject = %dlq_subject,
                        error = %e,
                        "failed to publish to DLQ"
                    );
                }
            }
        }

        warn!("DLQ advisory listener ended");
    }
}

fn dlq_message_id(
    advisory: &serde_json::Value,
    stream_name: &str,
    consumer_name: &str,
    stream_seq: u64,
) -> String {
    advisory
        .get("id")
        .and_then(|v| v.as_str())
        .filter(|id| !id.is_empty())
        .map(|id| format!("dlq-advisory:{}", id))
        .unwrap_or_else(|| {
            format!(
                "dlq-advisory:{}:{}:{}",
                stream_name, consumer_name, stream_seq
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_dlq_constants() {
        assert_eq!(DLQ_STREAM_NAME, "DEAD_LETTERS");
        assert_eq!(DLQ_RETENTION_SECS, 86400);
    }

    #[test]
    fn test_advisory_subject_pattern() {
        assert!(ADVISORY_SUBJECT.ends_with(">"));
        assert!(ADVISORY_SUBJECT.starts_with("$JS.EVENT.ADVISORY"));
    }

    /// Live JetStream test for the DLQ's durability reconcile
    /// (CodeRabbit findings on PR #3263).
    ///
    /// `get_or_create_stream` silently ignores the requested config when the
    /// stream already exists, so a dead-letter stream created before these
    /// knobs existed would keep memory storage with no signal. The reconcile
    /// must:
    ///
    /// 1. honour storage on the fresh-create path,
    /// 2. on an existing stream whose storage differs, NOT fail, and above all
    ///    NOT delete/recreate — the DLQ is the forensic record of already-lost
    ///    work, so destroying it to change its type would be the exact failure
    ///    this knob exists to prevent, and
    /// 3. reconcile `num_replicas`, which unlike storage IS changeable.
    ///
    /// Runs against a uniquely-named stream, never the real `DEAD_LETTERS`:
    /// that name is shared by every gateway replica and every concurrent test
    /// on the broker, so operating on it would be destructive far outside this
    /// test's scope — and tearing it down for a fixture would be the very
    /// delete-and-recreate the code under test refuses to do. This is what
    /// `ensure_dlq_stream` is parameterized for; the production name is
    /// covered by `test_dlq_constants`.
    ///
    /// NATS-gated, like the publisher's stream tests.
    #[tokio::test]
    async fn test_dlq_stream_storage_is_honoured_but_never_recreated() {
        let Ok(url) = std::env::var("NATS_URL") else {
            eprintln!("skipping: NATS_URL not set");
            return;
        };
        let client =
            match tokio::time::timeout(Duration::from_secs(2), async_nats::connect(&url)).await {
                Ok(Ok(c)) => c,
                _ => {
                    eprintln!("skipping: could not connect to NATS at {url}");
                    return;
                }
            };
        let context = async_nats::jetstream::new(client.clone());
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        // Unique name AND unique subject: JetStream rejects two streams whose
        // subjects overlap, so reusing `sie.dlq.>` would collide with a real
        // DEAD_LETTERS on the same broker.
        let stream_name = format!("DEAD_LETTERS_TEST_{nanos}");
        let subject = format!("sie.dlqtest.{nanos}.>");

        // Fresh-create path must honour the requested storage.
        DlqListener::ensure_dlq_stream(
            &context,
            &stream_name,
            &subject,
            jetstream::stream::StorageType::File,
            1,
        )
        .await
        .expect("ensure_dlq_stream (fresh create)");
        let created = context
            .get_stream(&stream_name)
            .await
            .expect("get DLQ stream")
            .info()
            .await
            .expect("DLQ stream info")
            .config
            .clone();
        assert_eq!(
            created.storage,
            jetstream::stream::StorageType::File,
            "DLQ must be created with the configured storage type"
        );
        assert_eq!(created.num_replicas, 1);

        // Existing stream with diverging storage: must succeed, warn, and
        // leave the stream (and its contents) intact.
        DlqListener::ensure_dlq_stream(
            &context,
            &stream_name,
            &subject,
            jetstream::stream::StorageType::Memory,
            1,
        )
        .await
        .expect("a DLQ storage mismatch must not fail startup");
        let after = context
            .get_stream(&stream_name)
            .await
            .expect("re-get DLQ stream")
            .info()
            .await
            .expect("DLQ stream info after mismatch")
            .config
            .clone();
        assert_eq!(
            after.storage,
            jetstream::stream::StorageType::File,
            "the DLQ must NOT be deleted/recreated to change its storage type"
        );
        assert_eq!(
            after.max_age,
            Duration::from_secs(DLQ_RETENTION_SECS),
            "DLQ retention must survive the reconcile"
        );

        // Replicas, unlike storage, ARE reconciled onto the existing stream.
        // This phase is what discriminates the fix: the storage assertions
        // above hold under the old `get_or_create_stream`-only code too
        // (leaving the stream alone is what it already did, just silently),
        // whereas a replica change is a real, observable `update_stream`.
        //
        // Note this single-node server happily reports R3 without a cluster
        // to back it — NATS does not reject the request, which is precisely
        // why the chart guards `streamReplicas > 1` against
        // `nats.config.cluster.enabled` rather than leaving it to the broker.
        DlqListener::ensure_dlq_stream(
            &context,
            &stream_name,
            &subject,
            jetstream::stream::StorageType::File,
            3,
        )
        .await
        .expect("DLQ replica reconcile");
        assert_eq!(
            context
                .get_stream(&stream_name)
                .await
                .expect("re-get DLQ stream")
                .info()
                .await
                .expect("DLQ stream info after replica reconcile")
                .config
                .num_replicas,
            3,
            "a DLQ replica divergence must be reconciled onto the existing \
             stream, not silently ignored"
        );

        // Only this test's own stream is removed.
        context
            .delete_stream(&stream_name)
            .await
            .expect("delete DLQ test stream");
    }

    #[test]
    fn test_dlq_subject_format() {
        // DLQ subjects use sie.dlq.{model_normalized}
        let model_normalized = "BAAI_bge-m3";
        let subject = format!("sie.dlq.{}", model_normalized);
        assert_eq!(subject, "sie.dlq.BAAI_bge-m3");
    }

    #[test]
    fn test_dlq_message_id_prefers_advisory_id() {
        let advisory = json!({"id": "abc-123"});
        assert_eq!(
            dlq_message_id(&advisory, "WORK_POOL_default", "consumer", 42),
            "dlq-advisory:abc-123"
        );
    }

    #[test]
    fn test_dlq_message_id_falls_back_to_stream_consumer_sequence() {
        let advisory = json!({});
        assert_eq!(
            dlq_message_id(&advisory, "WORK_POOL_default", "consumer", 42),
            "dlq-advisory:WORK_POOL_default:consumer:42"
        );
    }

    #[test]
    fn test_dlq_message_id_empty_string_falls_back_to_stream_consumer_sequence() {
        let advisory = json!({"id": ""});
        assert_eq!(
            dlq_message_id(&advisory, "WORK_POOL_default", "consumer", 42),
            "dlq-advisory:WORK_POOL_default:consumer:42"
        );
    }
}
