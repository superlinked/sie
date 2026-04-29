use std::time::Duration;

use async_nats::jetstream;
use futures_util::StreamExt;
use tracing::{error, info, warn};

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

        // Subscribe to max-delivery advisory events
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
            // `sie.work.{normalize_model_id(model)}.{pool}` — exactly four
            // dot-separated tokens, with the 3rd token already safe to use as
            // a single NATS token (no `.`, `*`, `>`, or whitespace).  We keep
            // the `/` -> `_` belt-and-suspenders replacement as a no-op for
            // correctly-normalized subjects and a fallback for any legacy
            // un-normalized messages still in-flight at the time of upgrade.
            let model_normalized = advisory
                .get("subject")
                .and_then(|v| v.as_str())
                .and_then(|s| {
                    let parts: Vec<&str> = s.splitn(4, '.').collect();
                    if parts.len() >= 3 {
                        Some(parts[2].replace('/', "_"))
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| format!("{}.{}.{}", stream_name, consumer_name, stream_seq));

            // Forward the advisory payload to the DLQ stream
            let dlq_subject = format!("sie.dlq.{}", model_normalized);

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
            let publish_result: Result<(), String> =
                match jetstream.publish(dlq_subject.clone(), payload.into()).await {
                    Ok(ack_future) => ack_future.await.map(|_| ()).map_err(|e| e.to_string()),
                    Err(e) => Err(e.to_string()),
                };
            match publish_result {
                Ok(()) => {
                    crate::metrics::DLQ_EVENTS
                        .with_label_values(&[stream_name, consumer_name])
                        .inc();
                    info!(
                        subject = %dlq_subject,
                        stream = %stream_name,
                        seq = stream_seq,
                        "forwarded dead letter to DLQ"
                    );
                }
                Err(e) => {
                    error!(
                        subject = %dlq_subject,
                        error = %e,
                        "failed to publish to DLQ"
                    );
                    crate::metrics::DLQ_REPUBLISH_FAILURES
                        .with_label_values(&[stream_name, consumer_name])
                        .inc();
                }
            }
        }

        warn!("DLQ advisory listener ended");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
