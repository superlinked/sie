//! MessagePack audio-envelope preparation shared by Rust ingress boundaries.
//!
//! Callers can either prepare canonical PCM or inspect the same encoded envelope
//! for exact duration without materializing PCM. Both
//! paths deliberately use this parser so field validation and declared-
//! metadata checks cannot drift.

use rmpv::Value;
use thiserror::Error;

use crate::{
    decode_audio_to_pcm16, inspect_audio, AudioLimits, AudioPrepError, InspectedAudio,
    PreparedPcm16,
};

#[derive(Debug, Error, PartialEq, Eq)]
pub enum PrepareAudioError {
    #[error("audio must be an object")]
    InvalidAudioObject,
    #[error("audio.{0} is required")]
    MissingField(&'static str),
    #[error("audio.{field} has an invalid value: {reason}")]
    InvalidField {
        field: &'static str,
        reason: &'static str,
    },
    #[error("audio contains duplicate field '{0}'")]
    DuplicateField(&'static str),
    #[error("audio contains an unknown field")]
    UnknownField,
    #[error("declared audio sample_rate {declared} does not match decoded {decoded} Hz")]
    SampleRateMismatch { declared: u32, decoded: u32 },
    #[error("{0}")]
    Decode(#[from] AudioPrepError),
}

#[derive(Debug)]
pub enum AudioPreparation {
    Ready(Value),
    Decode(Value),
}

struct EncodedAudioItem {
    item: Value,
    data: Vec<u8>,
    format: Option<String>,
    declared_sample_rate: Option<u32>,
}

enum ParsedAudioItem {
    Ready(Value),
    Decode(EncodedAudioItem),
}

/// Classify cheap pass-through inputs before acquiring a bounded decode permit.
/// Duplicate top-level audio fields are still rejected.
pub fn classify_item(mut item: Value) -> Result<AudioPreparation, PrepareAudioError> {
    let Value::Map(item_fields) = &mut item else {
        return Ok(AudioPreparation::Ready(item));
    };
    let mut audio_index = None;
    for (index, (key, _)) in item_fields.iter().enumerate() {
        if value_key_eq(key, "audio") {
            if audio_index.is_some() {
                return Err(PrepareAudioError::DuplicateField("audio"));
            }
            audio_index = Some(index);
        }
    }
    let Some(audio_index) = audio_index else {
        return Ok(AudioPreparation::Ready(item));
    };
    if item_fields[audio_index].1.is_nil() {
        item_fields.swap_remove(audio_index);
        return Ok(AudioPreparation::Ready(item));
    }
    Ok(AudioPreparation::Decode(item))
}

/// Replace encoded `item.audio` with canonical PCM metadata returned
/// separately. Items without audio pass through without allocation.
pub fn prepare_item(item: Value) -> Result<(Value, Option<PreparedPcm16>), PrepareAudioError> {
    let encoded = match parse_item(item)? {
        ParsedAudioItem::Ready(item) => return Ok((item, None)),
        ParsedAudioItem::Decode(encoded) => encoded,
    };
    let prepared = decode_audio_to_pcm16(
        encoded.data,
        encoded.format.as_deref(),
        AudioLimits::default(),
    )?;
    validate_declared_sample_rate(encoded.declared_sample_rate, prepared.source_sample_rate)?;
    debug_assert_eq!(prepared.pcm_s16le.len() as u64, prepared.sample_count * 2);
    Ok((encoded.item, Some(prepared)))
}

/// Validate an encoded audio item and derive exact decoded-source duration
/// without materializing canonical PCM. Items without audio return `None`.
pub fn inspect_item(item: Value) -> Result<Option<InspectedAudio>, PrepareAudioError> {
    let encoded = match parse_item(item)? {
        ParsedAudioItem::Ready(_) => return Ok(None),
        ParsedAudioItem::Decode(encoded) => encoded,
    };
    let inspected = inspect_audio(
        encoded.data,
        encoded.format.as_deref(),
        AudioLimits::default(),
    )?;
    validate_declared_sample_rate(encoded.declared_sample_rate, inspected.source_sample_rate)?;
    Ok(Some(inspected))
}

fn parse_item(item: Value) -> Result<ParsedAudioItem, PrepareAudioError> {
    let mut item = match classify_item(item)? {
        AudioPreparation::Ready(item) => return Ok(ParsedAudioItem::Ready(item)),
        AudioPreparation::Decode(item) => item,
    };
    let Value::Map(item_fields) = &mut item else {
        unreachable!("only map items with non-null audio require decoding");
    };
    let audio = take_unique(item_fields, "audio")?
        .expect("classification guarantees one non-null audio field");
    let Value::Map(mut audio_fields) = audio else {
        return Err(PrepareAudioError::InvalidAudioObject);
    };

    let data = match take_unique(&mut audio_fields, "data")? {
        Some(Value::Binary(data)) => data,
        Some(_) => {
            return Err(PrepareAudioError::InvalidField {
                field: "data",
                reason: "expected binary bytes",
            });
        }
        None => return Err(PrepareAudioError::MissingField("data")),
    };
    let format = match take_unique(&mut audio_fields, "format")? {
        None | Some(Value::Nil) => None,
        Some(Value::String(value)) => {
            let value = value.into_str().ok_or(PrepareAudioError::InvalidField {
                field: "format",
                reason: "expected valid UTF-8",
            })?;
            if value.len() > 32 {
                return Err(PrepareAudioError::InvalidField {
                    field: "format",
                    reason: "expected at most 32 UTF-8 bytes",
                });
            }
            Some(value)
        }
        Some(_) => {
            return Err(PrepareAudioError::InvalidField {
                field: "format",
                reason: "expected a string or null",
            });
        }
    };
    let declared_sample_rate = match take_unique(&mut audio_fields, "sample_rate")? {
        None | Some(Value::Nil) => None,
        Some(Value::Integer(value)) => Some(
            value
                .as_u64()
                .and_then(|rate| u32::try_from(rate).ok())
                .filter(|rate| *rate > 0)
                .ok_or(PrepareAudioError::InvalidField {
                    field: "sample_rate",
                    reason: "expected a positive 32-bit integer or null",
                })?,
        ),
        Some(_) => {
            return Err(PrepareAudioError::InvalidField {
                field: "sample_rate",
                reason: "expected a positive 32-bit integer or null",
            });
        }
    };
    if !audio_fields.is_empty() {
        return Err(PrepareAudioError::UnknownField);
    }

    Ok(ParsedAudioItem::Decode(EncodedAudioItem {
        item,
        data,
        format,
        declared_sample_rate,
    }))
}

fn validate_declared_sample_rate(
    declared_sample_rate: Option<u32>,
    decoded_sample_rate: u32,
) -> Result<(), PrepareAudioError> {
    if let Some(declared) = declared_sample_rate {
        if declared != decoded_sample_rate {
            return Err(PrepareAudioError::SampleRateMismatch {
                declared,
                decoded: decoded_sample_rate,
            });
        }
    }
    Ok(())
}

fn take_unique(
    fields: &mut Vec<(Value, Value)>,
    expected: &'static str,
) -> Result<Option<Value>, PrepareAudioError> {
    let mut found = None;
    for (index, (key, _)) in fields.iter().enumerate() {
        if value_key_eq(key, expected) {
            if found.is_some() {
                return Err(PrepareAudioError::DuplicateField(expected));
            }
            found = Some(index);
        }
    }
    Ok(found.map(|index| fields.swap_remove(index).1))
}

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

    fn field(key: &str, value: Value) -> (Value, Value) {
        (Value::String(key.into()), value)
    }

    fn wav_pcm16(frames: usize) -> Vec<u8> {
        let data_len = (frames * size_of::<i16>()) as u32;
        let mut wav = Vec::with_capacity(44 + data_len as usize);
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_len).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&16_000u32.to_le_bytes());
        wav.extend_from_slice(&32_000u32.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_len.to_le_bytes());
        wav.resize(44 + data_len as usize, 0);
        wav
    }

    fn mp3_silence(frame_count: usize) -> Vec<u8> {
        let mut frame = vec![0; 417];
        frame[..4].copy_from_slice(&[0xff, 0xfb, 0x90, 0xc4]);
        frame.repeat(frame_count)
    }

    fn audio_item(data: Vec<u8>, format: &str, sample_rate: u32) -> Value {
        Value::Map(vec![field(
            "audio",
            Value::Map(vec![
                field("data", Value::Binary(data)),
                field("format", Value::String(format.into())),
                field("sample_rate", Value::Integer(sample_rate.into())),
            ]),
        )])
    }

    #[test]
    fn inspection_and_preparation_share_envelope_and_source_metadata() {
        let cases = [
            Value::Map(vec![field("text", Value::String("hello".into()))]),
            Value::Map(vec![field("audio", Value::Nil)]),
            audio_item(wav_pcm16(1_601), "wav", 16_000),
            audio_item(mp3_silence(4), "mp3", 44_100),
        ];
        for item in cases {
            let inspected = inspect_item(item.clone()).unwrap();
            let (_, prepared) = prepare_item(item).unwrap();
            assert_eq!(
                inspected.as_ref().map(|value| (
                    value.duration_ms,
                    value.source_sample_rate,
                    value.source_channels,
                    value.source_sample_count,
                    value.container,
                )),
                prepared.as_ref().map(|value| (
                    value.duration_ms,
                    value.source_sample_rate,
                    value.source_channels,
                    value.source_sample_count,
                    value.container,
                )),
            );
        }
    }

    #[test]
    fn request_controlled_schema_strings_do_not_expand_errors() {
        let large_key = "x".repeat(1024 * 1024);
        let unknown = Value::Map(vec![field(
            "audio",
            Value::Map(vec![
                field("data", Value::Binary(wav_pcm16(16))),
                (Value::String(large_key.into()), Value::Nil),
            ]),
        )]);
        let unknown_error = inspect_item(unknown).unwrap_err();
        assert_eq!(unknown_error.to_string(), "audio contains an unknown field");

        let large_format = "y".repeat(1024 * 1024);
        let format_error =
            inspect_item(audio_item(wav_pcm16(16), &large_format, 16_000)).unwrap_err();
        assert_eq!(
            format_error.to_string(),
            "audio.format has an invalid value: expected at most 32 UTF-8 bytes",
        );

        let mismatch = inspect_item(audio_item(wav_pcm16(16), "mp3", 16_000)).unwrap_err();
        assert_eq!(
            mismatch.to_string(),
            "declared audio format does not match probed container",
        );
    }

    #[test]
    fn inspection_and_preparation_share_envelope_errors() {
        let cases = [
            Value::Map(vec![field(
                "audio",
                Value::Map(vec![field("data", Value::String("bytes".into()))]),
            )]),
            Value::Map(vec![field(
                "audio",
                Value::Map(vec![
                    field("data", Value::Binary(wav_pcm16(16))),
                    field("unknown", Value::Nil),
                ]),
            )]),
            audio_item(wav_pcm16(16), "wav", 8_000),
            audio_item(b"\xff\xfb\x90\xc4".to_vec(), "mp3", 44_100),
        ];
        for item in cases {
            assert_eq!(
                inspect_item(item.clone()).unwrap_err(),
                prepare_item(item).unwrap_err(),
            );
        }
    }
}
