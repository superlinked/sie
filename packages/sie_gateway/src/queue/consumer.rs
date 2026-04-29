use std::sync::Arc;
use std::time::Duration;

use async_nats::jetstream;
use async_nats::jetstream::consumer::pull::Config as ConsumerConfig;
use futures_util::StreamExt;
use tracing::{debug, error, info, warn};

use super::publisher::{WorkItem, WorkResult};

#[allow(dead_code)]
const ACK_WAIT_SECS: u64 = 30;
#[allow(dead_code)]
const MAX_DELIVER: i64 = 3;
#[allow(dead_code)]
const MAX_ACK_PENDING: i64 = 1000;

/// WorkConsumer pulls work items from a JetStream stream and processes them
/// via a callback. Results are published back to the reply inbox.
#[allow(dead_code)]
pub struct WorkConsumer {
    jetstream: jetstream::Context,
    client: async_nats::Client,
    pool: String,
    bundle: String,
}

#[allow(dead_code)]
impl WorkConsumer {
    pub fn new(
        jetstream: jetstream::Context,
        client: async_nats::Client,
        pool: &str,
        bundle: &str,
    ) -> Self {
        Self {
            jetstream,
            client,
            pool: pool.to_string(),
            bundle: bundle.to_string(),
        }
    }

    fn consumer_name(&self) -> String {
        format!("WORK_CONSUMER_{}_{}", self.bundle, self.pool)
    }

    /// Create or get a durable pull consumer for the given pool.
    pub async fn ensure_consumer(&self) -> Result<(), String> {
        let stream_name = format!("WORK_POOL_{}", self.pool);
        let consumer_name = self.consumer_name();

        let stream = self
            .jetstream
            .get_stream(&stream_name)
            .await
            .map_err(|e| format!("get stream {}: {}", stream_name, e))?;

        stream
            .get_or_create_consumer(
                &consumer_name,
                ConsumerConfig {
                    durable_name: Some(consumer_name.clone()),
                    ack_wait: Duration::from_secs(ACK_WAIT_SECS),
                    max_deliver: MAX_DELIVER,
                    max_ack_pending: MAX_ACK_PENDING,
                    ..Default::default()
                },
            )
            .await
            .map_err(|e| format!("create consumer {}: {}", consumer_name, e))?;

        info!(
            stream = %stream_name,
            consumer = %consumer_name,
            "ensured durable pull consumer"
        );

        Ok(())
    }

    /// Start consuming work items. The `handler` callback processes each item
    /// and returns a WorkResult.
    pub async fn start<F, Fut>(self: Arc<Self>, handler: F) -> Result<(), String>
    where
        F: Fn(WorkItem) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = WorkResult> + Send + 'static,
    {
        let stream_name = format!("WORK_POOL_{}", self.pool);
        let consumer_name = self.consumer_name();

        let stream = self
            .jetstream
            .get_stream(&stream_name)
            .await
            .map_err(|e| format!("get stream {}: {}", stream_name, e))?;

        let consumer = stream
            .get_consumer::<ConsumerConfig>(&consumer_name)
            .await
            .map_err(|e| format!("get consumer {}: {}", consumer_name, e))?;

        let handler = Arc::new(handler);
        let client = self.client.clone();

        tokio::spawn(async move {
            info!(
                stream = %stream_name,
                consumer = %consumer_name,
                "starting work consumer loop"
            );

            let mut messages = match consumer.messages().await {
                Ok(m) => m,
                Err(e) => {
                    error!(error = %e, "failed to get message stream");
                    return;
                }
            };

            while let Some(msg_result) = messages.next().await {
                let msg = match msg_result {
                    Ok(m) => m,
                    Err(e) => {
                        warn!(error = %e, "error receiving message");
                        continue;
                    }
                };

                // Decode work item
                let work_item: WorkItem = match rmp_serde::from_slice(&msg.payload) {
                    Ok(item) => item,
                    Err(e) => {
                        warn!(error = %e, "failed to decode work item");
                        if let Err(e) = msg.ack().await {
                            warn!(error = %e, "failed to ack malformed message");
                        }
                        continue;
                    }
                };

                let reply_inbox = work_item.reply_subject.clone();
                let request_id = work_item.request_id.clone();
                let item_index = work_item.item_index;

                debug!(
                    request_id = %request_id,
                    item_index = item_index,
                    operation = %work_item.operation,
                    "processing work item"
                );

                // Process the work item
                let result = handler(work_item).await;

                // Acknowledge the message (explicit ACK)
                if let Err(e) = msg.ack().await {
                    warn!(
                        request_id = %request_id,
                        error = %e,
                        "failed to ack message"
                    );
                }

                // Publish result back to reply inbox
                let encoded = match rmp_serde::to_vec(&result) {
                    Ok(bytes) => bytes,
                    Err(e) => {
                        error!(
                            request_id = %request_id,
                            error = %e,
                            "failed to encode result"
                        );
                        continue;
                    }
                };

                if let Err(e) = client.publish(reply_inbox.clone(), encoded.into()).await {
                    error!(
                        request_id = %request_id,
                        reply_inbox = %reply_inbox,
                        error = %e,
                        "failed to publish result"
                    );
                }
            }

            warn!("work consumer loop ended");
        });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ack_wait_duration() {
        assert_eq!(ACK_WAIT_SECS, 30);
    }

    #[test]
    fn test_max_deliver() {
        assert_eq!(MAX_DELIVER, 3);
    }

    #[test]
    fn test_max_ack_pending() {
        assert_eq!(MAX_ACK_PENDING, 1000);
    }

    #[test]
    fn test_stream_name_format() {
        let pool = "default";
        let expected = "WORK_POOL_default";
        assert_eq!(format!("WORK_POOL_{}", pool), expected);
    }

    #[test]
    fn test_consumer_name_format() {
        assert_eq!(
            format!("WORK_CONSUMER_{}_{}", "default", "eval-l4"),
            "WORK_CONSUMER_default_eval-l4"
        );
        assert_eq!(
            format!("WORK_CONSUMER_{}_{}", "my-bundle", "default"),
            "WORK_CONSUMER_my-bundle_default"
        );
    }
}
