//! Audio preparation at the worker's Rust trust boundary.
//!
//! Encoded media is removed from the user item before canonical PCM is attached
//! to the Python IPC request. The model adapter therefore never receives a
//! compressed payload and does not own container decoding or resampling.

#[cfg(test)]
use rmpv::Value;

use crate::ipc_types::{PreparedAudioPcm16, WireValue};

pub use sie_audio_prep::msgpack::{AudioPreparation, PrepareAudioError};

pub const MAX_AUDIO_BATCH_DURATION_MS: u64 = sie_audio_prep::DEFAULT_MAX_DURATION_MS;

/// Classify cheap pass-through inputs before the dispatcher acquires the
/// bounded decode permit. Duplicate top-level audio fields are still rejected.
pub fn classify_item(item: WireValue) -> Result<AudioPreparation, PrepareAudioError> {
    sie_audio_prep::msgpack::classify_item(item)
}

/// Replace an encoded `item.audio` value with canonical sidecar-prepared PCM.
/// Items without audio pass through without allocation.
pub fn prepare_item(
    item: WireValue,
) -> Result<(WireValue, Option<PreparedAudioPcm16>), PrepareAudioError> {
    let (item, prepared) = sie_audio_prep::msgpack::prepare_item(item)?;
    Ok((
        item,
        prepared.map(|prepared| PreparedAudioPcm16 {
            pcm_s16le: prepared.pcm_s16le,
            sample_rate: prepared.sample_rate,
            sample_count: prepared.sample_count,
            source_sample_count: prepared.source_sample_count,
            duration_ms: prepared.duration_ms,
            source_sample_rate: prepared.source_sample_rate,
            source_channels: prepared.source_channels as u32,
            container: prepared.container.canonical_name().to_string(),
        }),
    ))
}

#[cfg(test)]
fn value_key_eq(value: &Value, expected: &str) -> bool {
    match value {
        Value::String(value) => value.as_str() == Some(expected),
        Value::Binary(value) => std::str::from_utf8(value).ok() == Some(expected),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav_pcm16(frames: usize) -> Vec<u8> {
        let data_len = (frames * size_of::<i16>()) as u32;
        let mut wav = Vec::with_capacity(44 + data_len as usize);
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_len).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16_u32.to_le_bytes());
        wav.extend_from_slice(&1_u16.to_le_bytes());
        wav.extend_from_slice(&1_u16.to_le_bytes());
        wav.extend_from_slice(&16_000_u32.to_le_bytes());
        wav.extend_from_slice(&32_000_u32.to_le_bytes());
        wav.extend_from_slice(&2_u16.to_le_bytes());
        wav.extend_from_slice(&16_u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_len.to_le_bytes());
        for _ in 0..frames {
            wav.extend_from_slice(&0_i16.to_le_bytes());
        }
        wav
    }

    fn field(name: &str, value: Value) -> (Value, Value) {
        (Value::from(name), value)
    }

    #[test]
    fn prepares_pcm_and_removes_encoded_audio_from_ipc_item() {
        let encoded = wav_pcm16(1_601);
        let item = Value::Map(vec![
            field("text", Value::from("metadata can remain")),
            field(
                "audio",
                Value::Map(vec![
                    field("data", Value::Binary(encoded)),
                    field("format", Value::from("wav")),
                    field("sample_rate", Value::from(16_000)),
                ]),
            ),
        ]);

        let (item, prepared) = prepare_item(item).unwrap();
        let prepared = prepared.expect("audio should be prepared");
        assert_eq!(prepared.sample_rate, 16_000);
        assert_eq!(prepared.sample_count, 1_601);
        assert_eq!(prepared.duration_ms, 101);
        assert_eq!(prepared.source_sample_count, 1_601);
        assert_eq!(prepared.pcm_s16le.len(), 1_601 * 2);

        let Value::Map(fields) = &item else {
            panic!("prepared item should stay a map");
        };
        assert!(fields.iter().any(|(key, _)| value_key_eq(key, "text")));
        assert!(!fields.iter().any(|(key, _)| value_key_eq(key, "audio")));
        let wire = rmp_serde::to_vec_named(&(item, prepared)).unwrap();
        assert!(!wire.windows(4).any(|window| window == b"RIFF"));
    }

    #[test]
    fn inline_and_offloaded_msgpack_audio_prepare_identically() {
        let inline = Value::Map(vec![
            field("text", Value::from("metadata can remain")),
            field(
                "audio",
                Value::Map(vec![
                    field("data", Value::Binary(wav_pcm16(1_601))),
                    field("format", Value::from("wav")),
                ]),
            ),
        ]);
        let payload = rmp_serde::to_vec_named(&inline).expect("offloaded item should encode");
        let offloaded: Value =
            rmp_serde::from_slice(&payload).expect("offloaded item should decode");

        let inline_prepared = prepare_item(inline).expect("inline audio should prepare");
        let offloaded_prepared = prepare_item(offloaded).expect("offloaded audio should prepare");
        assert_eq!(
            rmp_serde::to_vec_named(&inline_prepared).unwrap(),
            rmp_serde::to_vec_named(&offloaded_prepared).unwrap(),
        );
    }

    #[test]
    fn preserves_ceil_duration_for_submillisecond_audio() {
        let item = Value::Map(vec![field(
            "audio",
            Value::Map(vec![field("data", Value::Binary(wav_pcm16(1)))]),
        )]);

        let (_, prepared) = prepare_item(item).unwrap();
        let prepared = prepared.expect("audio should be prepared");
        assert_eq!(prepared.source_sample_count, 1);
        assert_eq!(prepared.duration_ms, 1);
    }

    #[test]
    fn rejects_declared_metadata_that_disagrees_with_decoded_audio() {
        let sample_rate_error = prepare_item(Value::Map(vec![field(
            "audio",
            Value::Map(vec![
                field("data", Value::Binary(wav_pcm16(16))),
                field("sample_rate", Value::from(44_100)),
            ]),
        )]))
        .unwrap_err();
        assert!(matches!(
            sample_rate_error,
            PrepareAudioError::SampleRateMismatch {
                declared: 44_100,
                decoded: 16_000
            }
        ));

        let format_error = prepare_item(Value::Map(vec![field(
            "audio",
            Value::Map(vec![
                field("data", Value::Binary(wav_pcm16(16))),
                field("format", Value::from("mp3")),
            ]),
        )]))
        .unwrap_err();
        assert!(matches!(
            format_error,
            PrepareAudioError::Decode(sie_audio_prep::AudioPrepError::FormatMismatch)
        ));
    }

    #[test]
    fn rejects_duplicate_and_unknown_audio_fields() {
        let duplicate = prepare_item(Value::Map(vec![field(
            "audio",
            Value::Map(vec![
                field("data", Value::Binary(wav_pcm16(16))),
                field("format", Value::from("wav")),
                field("format", Value::from("wav")),
            ]),
        )]))
        .unwrap_err();
        assert!(matches!(
            duplicate,
            PrepareAudioError::DuplicateField("format")
        ));

        let unknown = prepare_item(Value::Map(vec![field(
            "audio",
            Value::Map(vec![
                field("data", Value::Binary(wav_pcm16(16))),
                field("mystery", Value::Nil),
            ]),
        )]))
        .unwrap_err();
        assert!(matches!(unknown, PrepareAudioError::UnknownField));
    }

    #[test]
    fn classifies_absent_and_null_audio_without_decoding() {
        let absent = Value::Map(vec![field("text", Value::from("pass through"))]);
        let AudioPreparation::Ready(absent) = classify_item(absent).unwrap() else {
            panic!("absent audio should be ready");
        };
        assert!(matches!(absent, Value::Map(fields) if fields.len() == 1));

        let null = Value::Map(vec![
            field("text", Value::from("pass through")),
            field("audio", Value::Nil),
        ]);
        let AudioPreparation::Ready(Value::Map(fields)) = classify_item(null).unwrap() else {
            panic!("null audio should be ready");
        };
        assert!(!fields.iter().any(|(key, _)| value_key_eq(key, "audio")));
    }

    #[test]
    fn classification_rejects_duplicate_top_level_audio() {
        let error = classify_item(Value::Map(vec![
            field("audio", Value::Nil),
            field("audio", Value::Nil),
        ]))
        .unwrap_err();
        assert!(matches!(error, PrepareAudioError::DuplicateField("audio")));
    }
}
