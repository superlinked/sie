//! NATS wire types — these MUST stay in lockstep with
//! `packages/sie_gateway/src/queue/publisher.rs` (the publisher). All
//! fields use `#[serde(default)]` so a forward-compatible producer can
//! add fields without breaking us.
//!
//! Wire format is msgpack **named** (`rmp_serde::to_vec_named` / decoded
//! from a msgpack map into the named struct).

use std::fmt;

use serde::de::{self, IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};

/// User-supplied item payload carried as msgpack. Do not convert this to
/// `serde_json::Value`: msgpack `bin` / `ext` fields are valid for documents,
/// images, and numpy payloads, but JSON has no representation for them.
pub type WireValue = rmpv::Value;

/// Work item pulled from JetStream. Must stay wire-compatible with the
/// gateway publisher.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkItem {
    pub work_item_id: String,
    pub request_id: String,
    pub item_index: u32,
    pub total_items: u32,
    pub operation: String,
    pub model_id: String,
    #[serde(default)]
    pub profile_id: String,
    /// Mirror of `WorkItem.engine` from the gateway publisher. The
    /// gateway started populating this in the engine-routing follow-up
    /// (today only `pytorch` is in use); older gateway builds omit it
    /// and we decode them as the empty string. Workers can use it for
    /// an optional sanity check (e.g. WARN if the dispatched engine
    /// doesn't match the worker's configured backend) — for the
    /// pre-engine fleet behaviour is unchanged.
    #[serde(default)]
    pub engine: String,
    pub pool_name: String,
    #[serde(default)]
    pub admission_pool: String,
    pub machine_profile: String,
    #[serde(default)]
    pub item: Option<WireValue>,
    #[serde(default)]
    pub payload_ref: Option<String>,
    #[serde(default)]
    pub output_types: Option<Vec<String>>,
    #[serde(default)]
    pub instruction: Option<String>,
    #[serde(default)]
    pub is_query: bool,
    #[serde(default)]
    pub options: Option<serde_json::Value>,
    #[serde(default)]
    pub query_item: Option<WireValue>,
    #[serde(default)]
    pub query_payload_ref: Option<String>,
    #[serde(default)]
    pub score_items: Option<Vec<WireValue>>,
    #[serde(default)]
    pub labels: Option<Vec<String>>,
    #[serde(default)]
    pub output_schema: Option<serde_json::Value>,
    #[serde(default)]
    /// Generation parameters use the same msgpack-native value as media
    /// items. Legacy producers still serialize ordinary JSON maps, while
    /// sidecar-prepared vision requests can carry image data as msgpack
    /// binary without a second base64 allocation in Python.
    pub generate: Option<WireValue>,
    #[serde(default)]
    pub routing_key: Option<String>,
    #[serde(default)]
    pub prompt_cache_key: Option<String>,
    #[serde(default)]
    pub bundle_config_hash: String,
    #[serde(default)]
    pub router_id: String,
    /// Result-transport capability negotiated by the gateway. Older
    /// gateways omit this field and therefore retain the one-message
    /// `WorkResult` / compact `PAYLOAD_TOO_LARGE` behavior.
    #[serde(default)]
    pub accepts_result_chunks: bool,
    pub reply_subject: String,
    #[serde(default)]
    pub traceparent: Option<String>,
    #[serde(default)]
    pub tracestate: Option<String>,
    #[serde(default)]
    pub timestamp: f64,
    /// Absolute Unix-epoch seconds on the gateway clock after which no caller
    /// waits for this item. Older gateways omit it and the item is unbounded.
    /// A non-numeric value is treated as absent rather than making the whole
    /// item undecodable.
    #[serde(
        default,
        deserialize_with = "deserialize_lenient_seconds",
        skip_serializing_if = "Option::is_none"
    )]
    pub deadline: Option<f64>,
    /// Set by the gateway when it sent the item to a remote profile in place
    /// of a local route that refused it; names the refusal. Someone is
    /// waiting to answer with that local refusal, so backend NakRetry outcomes
    /// are published as refusals and ACKed after successful publication.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback_reason: Option<String>,
}

fn deserialize_lenient_seconds<'de, D>(deserializer: D) -> Result<Option<f64>, D::Error>
where
    D: Deserializer<'de>,
{
    struct LenientSeconds;

    impl<'de> Visitor<'de> for LenientSeconds {
        type Value = Option<f64>;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("a number of seconds; any other value is treated as absent")
        }

        fn visit_f64<E: de::Error>(self, value: f64) -> Result<Self::Value, E> {
            Ok(Some(value))
        }

        fn visit_i64<E: de::Error>(self, value: i64) -> Result<Self::Value, E> {
            Ok(Some(value as f64))
        }

        fn visit_u64<E: de::Error>(self, value: u64) -> Result<Self::Value, E> {
            Ok(Some(value as f64))
        }

        fn visit_bool<E: de::Error>(self, _: bool) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_str<E: de::Error>(self, _: &str) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_bytes<E: de::Error>(self, _: &[u8]) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_some<D2: Deserializer<'de>>(self, inner: D2) -> Result<Self::Value, D2::Error> {
            inner.deserialize_any(self)
        }

        fn visit_newtype_struct<D2: Deserializer<'de>>(
            self,
            inner: D2,
        ) -> Result<Self::Value, D2::Error> {
            IgnoredAny::deserialize(inner)?;
            Ok(None)
        }

        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
            while seq.next_element::<IgnoredAny>()?.is_some() {}
            Ok(None)
        }

        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
            while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
            Ok(None)
        }
    }

    deserializer.deserialize_any(LenientSeconds)
}

/// Per-item result published back to the gateway's inbox.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkResult {
    #[serde(default)]
    pub work_item_id: String,
    pub request_id: String,
    #[serde(default)]
    pub item_index: u32,
    #[serde(default)]
    pub success: bool,
    #[serde(default, with = "serde_bytes")]
    pub result_msgpack: Vec<u8>,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub error_code: Option<String>,
    #[serde(default)]
    pub inference_ms: Option<f64>,
    #[serde(default)]
    pub queue_ms: Option<f64>,
    #[serde(default)]
    pub processing_ms: Option<f64>,
    #[serde(default)]
    pub worker_id: Option<String>,
    #[serde(default)]
    pub tokenization_ms: Option<f64>,
    #[serde(default)]
    pub postprocessing_ms: Option<f64>,
    #[serde(default)]
    pub payload_fetch_ms: Option<f64>,
    /// Authoritative billable-unit counts passed through from the engine's
    /// `ItemOutcome.units` (P3.5, design §7.3). `skip_serializing_if` keeps
    /// legacy consumers' maps unchanged when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub units: Option<crate::ipc_types::UnitCounts>,
    #[serde(default)]
    pub worker_direct: bool,
    /// Bundle hash held stable by the execution barrier while this item ran.
    /// The gateway uses it as post-execution provenance evidence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub executed_bundle_config_hash: Option<String>,
    /// Seconds after which a retryable error may succeed, passed through from
    /// the engine's `ItemOutcome.retry_after_s`. Absent on success and when
    /// the engine gave no hint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_after_s: Option<u32>,
}

/// One bounded chunk of a serialized [`WorkResult`].
///
/// This is a named-msgpack envelope published only when the originating
/// [`WorkItem`] explicitly negotiated chunk support. `payload` contains a
/// byte slice of the *complete, already serialized* `WorkResult`; consumers
/// must reassemble all chunks and verify `transfer_digest` before decoding it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResultChunkV1 {
    pub kind: String,
    pub work_item_id: String,
    pub request_id: String,
    pub item_index: u32,
    #[serde(with = "serde_bytes")]
    pub transfer_digest: Vec<u8>,
    pub chunk_index: u32,
    pub chunk_count: u32,
    pub total_bytes: u64,
    #[serde(with = "serde_bytes")]
    pub payload: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg_value(value: serde_json::Value) -> WireValue {
        let bytes = rmp_serde::to_vec_named(&value).unwrap();
        rmp_serde::from_slice(&bytes).unwrap()
    }

    fn sample_work_item() -> WorkItem {
        WorkItem {
            work_item_id: "req-1.0".into(),
            request_id: "req-1".into(),
            item_index: 0,
            total_items: 1,
            operation: "encode".into(),
            model_id: "BAAI/bge-m3".into(),
            profile_id: String::new(),
            engine: String::new(),
            pool_name: "l4".into(),
            admission_pool: "l4".into(),
            machine_profile: "l4-spot".into(),
            item: Some(msg_value(serde_json::json!({"text": "hello"}))),
            payload_ref: None,
            output_types: Some(vec!["dense".into()]),
            instruction: None,
            is_query: false,
            options: None,
            query_item: None,
            query_payload_ref: None,
            score_items: None,
            labels: None,
            output_schema: None,
            generate: None,
            routing_key: None,
            prompt_cache_key: None,
            bundle_config_hash: "abc".into(),
            router_id: "r1".into(),
            accepts_result_chunks: false,
            reply_subject: "_INBOX.r1.req-1".into(),
            traceparent: None,
            tracestate: None,
            timestamp: 1_700_000_000.0,
            deadline: None,
            fallback_reason: None,
        }
    }

    #[test]
    fn work_item_msgpack_named_roundtrip() {
        let item = sample_work_item();
        let bytes = rmp_serde::to_vec_named(&item).unwrap();
        let back: WorkItem = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(back.request_id, "req-1");
        assert_eq!(back.item_index, 0);
        assert_eq!(back.total_items, 1);
        assert_eq!(back.operation, "encode");
        assert_eq!(back.model_id, "BAAI/bge-m3");
        assert_eq!(back.output_types, Some(vec!["dense".to_string()]));
        assert_eq!(
            back.item,
            Some(msg_value(serde_json::json!({"text": "hello"})))
        );
    }

    #[test]
    fn work_item_forward_compatible_extra_field_ignored() {
        // Producer adds a new field we don't know about — we must still decode.
        let mut map = serde_json::to_value(sample_work_item()).unwrap();
        map["unknown_future_field"] = serde_json::json!(42);
        let bytes = rmp_serde::to_vec_named(&map).unwrap();
        let back: WorkItem = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(back.request_id, "req-1");
    }

    #[test]
    fn work_item_deadline_decodes_numbers_and_treats_other_values_as_absent() {
        let decode = |deadline: Option<serde_json::Value>| {
            let mut map = serde_json::to_value(sample_work_item()).unwrap();
            let object = map.as_object_mut().expect("work item map");
            object.remove("deadline");
            if let Some(deadline) = deadline {
                object.insert("deadline".into(), deadline);
            }
            let bytes = rmp_serde::to_vec_named(&map).unwrap();
            let item: WorkItem = rmp_serde::from_slice(&bytes)
                .expect("a malformed deadline must not make the item undecodable");
            assert_eq!(item.request_id, "req-1");
            item.deadline
        };

        assert_eq!(decode(None), None);
        assert_eq!(
            decode(Some(serde_json::json!(1_700_000_120.5))),
            Some(1_700_000_120.5)
        );
        assert_eq!(
            decode(Some(serde_json::json!(1_700_000_120))),
            Some(1_700_000_120.0)
        );
        assert_eq!(decode(Some(serde_json::json!(-3))), Some(-3.0));
        for other in [
            serde_json::json!("1700000120"),
            serde_json::json!(true),
            serde_json::Value::Null,
            serde_json::json!([1, 2]),
            serde_json::json!({"seconds": 1}),
        ] {
            assert_eq!(decode(Some(other.clone())), None, "{other}");
        }

        let mut item = sample_work_item();
        item.deadline = Some(1_700_000_120.5);
        let bytes = rmp_serde::to_vec_named(&item).unwrap();
        let back: WorkItem = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(back.deadline, Some(1_700_000_120.5));
    }

    #[test]
    fn work_item_missing_result_chunk_capability_defaults_false() {
        let mut map = serde_json::to_value(sample_work_item()).unwrap();
        map.as_object_mut()
            .expect("work item map")
            .remove("accepts_result_chunks");

        let bytes = rmp_serde::to_vec_named(&map).unwrap();
        let back: WorkItem = rmp_serde::from_slice(&bytes).unwrap();

        assert!(!back.accepts_result_chunks);
    }

    #[test]
    fn result_chunk_roundtrip_uses_raw_byte_fields() {
        let chunk = ResultChunkV1 {
            kind: "result_chunk_v1".into(),
            work_item_id: "req-1.0".into(),
            request_id: "req-1".into(),
            item_index: 0,
            transfer_digest: vec![7; 32],
            chunk_index: 1,
            chunk_count: 3,
            total_bytes: 1_234,
            payload: vec![0, 1, 2, 255],
        };

        let bytes = rmp_serde::to_vec_named(&chunk).unwrap();
        let back: ResultChunkV1 = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(back, chunk);

        let value: rmpv::Value = rmp_serde::from_slice(&bytes).unwrap();
        let rmpv::Value::Map(fields) = value else {
            panic!("chunk must use a named msgpack map");
        };
        for field in ["transfer_digest", "payload"] {
            let (_, value) = fields
                .iter()
                .find(|(key, _)| key.as_str() == Some(field))
                .unwrap_or_else(|| panic!("missing {field}"));
            assert!(
                matches!(value, rmpv::Value::Binary(_)),
                "{field} must be bin"
            );
        }
    }

    #[test]
    fn work_item_with_payload_ref_has_no_item() {
        let mut item = sample_work_item();
        item.item = None;
        item.payload_ref = Some("req-1_0.bin".into());
        let bytes = rmp_serde::to_vec_named(&item).unwrap();
        let back: WorkItem = rmp_serde::from_slice(&bytes).unwrap();
        assert!(back.item.is_none());
        assert_eq!(back.payload_ref, Some("req-1_0.bin".into()));
    }

    #[test]
    fn work_item_preserves_document_bytes() {
        let mut item = sample_work_item();
        let pdf_bytes = b"%PDF-1.4 tiny".to_vec();
        item.operation = "extract".into();
        item.model_id = "docling".into();
        item.item = Some(WireValue::Map(vec![(
            WireValue::from("document"),
            WireValue::Map(vec![
                (
                    WireValue::from("data"),
                    WireValue::Binary(pdf_bytes.clone()),
                ),
                (WireValue::from("format"), WireValue::from("pdf")),
            ]),
        )]));

        let bytes = rmp_serde::to_vec_named(&item).unwrap();
        let back: WorkItem = rmp_serde::from_slice(&bytes).unwrap();
        let Some(WireValue::Map(fields)) = back.item else {
            panic!("item should decode as msgpack map");
        };
        let Some((_, WireValue::Map(document))) = fields
            .iter()
            .find(|(key, _)| matches!(key, WireValue::String(s) if s.as_str() == Some("document")))
        else {
            panic!("document should decode as msgpack map");
        };
        let data = document
            .iter()
            .find_map(|(key, value)| {
                if matches!(key, WireValue::String(s) if s.as_str() == Some("data")) {
                    Some(value)
                } else {
                    None
                }
            })
            .expect("document.data");
        assert_eq!(data, &WireValue::Binary(pdf_bytes));
    }

    #[test]
    fn work_result_roundtrip_named() {
        let result = WorkResult {
            work_item_id: "req-1.2".into(),
            request_id: "req-1".into(),
            item_index: 2,
            success: true,
            result_msgpack: vec![0x81, 0xa5, b'h', b'e', b'l', b'l', b'o', 0x05],
            error: None,
            error_code: None,
            inference_ms: Some(12.5),
            queue_ms: None,
            processing_ms: Some(14.0),
            worker_id: Some("w-1".into()),
            tokenization_ms: None,
            postprocessing_ms: None,
            payload_fetch_ms: None,
            units: None,
            worker_direct: true,
            executed_bundle_config_hash: None,
            retry_after_s: None,
        };
        let bytes = rmp_serde::to_vec_named(&result).unwrap();
        let back: WorkResult = rmp_serde::from_slice(&bytes).unwrap();
        assert!(back.success);
        assert_eq!(back.request_id, "req-1");
        assert_eq!(back.item_index, 2);
        assert_eq!(back.result_msgpack.len(), 8);
        assert_eq!(back.inference_ms, Some(12.5));
        assert!(back.worker_direct);
    }
}
