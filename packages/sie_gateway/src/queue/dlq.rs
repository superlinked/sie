use std::time::Duration;

use async_nats::jetstream;
use futures_util::StreamExt;
use tracing::{debug, error, info, warn};

use crate::observability::metrics::{self as telemetry, QueueEvent, QueueEventOutcome};

const DLQ_STREAM_NAME: &str = "DEAD_LETTERS";
const DLQ_SUBJECT: &str = "sie.dlq.>";
const DLQ_RETENTION_SECS: u64 = 86400; // 24 hours
const ADVISORY_SUBJECT: &str = "$JS.EVENT.ADVISORY.CONSUMER.MAX_DELIVERIES.>";

pub struct DlqListener;

impl DlqListener {
    /// Start listening for NATS JetStream advisory events and routing
    /// max-delivery messages to a dead letter stream.
    pub async fn start(
        jetstream: jetstream::Context,
        client: async_nats::Client,
    ) -> Result<(), String> {
        // Ensure the dead letter stream exists
        jetstream
            .get_or_create_stream(jetstream::stream::Config {
                name: DLQ_STREAM_NAME.to_string(),
                subjects: vec![DLQ_SUBJECT.to_string()],
                retention: jetstream::stream::RetentionPolicy::Limits,
                storage: jetstream::stream::StorageType::Memory,
                max_age: Duration::from_secs(DLQ_RETENTION_SECS),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("create DLQ stream: {}", e))?;

        info!(
            stream = DLQ_STREAM_NAME,
            retention_hours = DLQ_RETENTION_SECS / 3600,
            "dead letter queue stream ready"
        );

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
