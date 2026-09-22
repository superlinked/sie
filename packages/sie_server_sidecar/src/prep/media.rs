//! Queue-side validation for msgpack-native media inputs.
//!
//! This is deliberately limited to transport-generic work: validate the
//! item/media shape, preserve binary payloads without copying, and enforce
//! ingress-equivalent count/byte limits before Python IPC. Compressed image
//! decode and document rendering remain adapter/engine-owned because the IPC
//! contract has no reusable decoded-media representation.

use base64::Engine as _;
use rmpv::Value;
use thiserror::Error;

/// The native encode/score/extract ingress body cap in the gateway.
///
/// A single media payload cannot legitimately exceed the complete request
/// that carried it. Rechecking the same upper bound here protects local-ingest
/// and direct-NATS paths that do not traverse HTTP ingress.
pub const MAX_MEDIA_BYTES_PER_ITEM: usize = 16 * 1024 * 1024;

/// Mirror of the gateway's widest accepted native extract request.
///
/// Offloaded items are re-encoded as msgpack after HTTP parsing, with binary
/// media retained as msgpack `bin` values rather than base64 strings.
pub const MAX_EXTRACT_REQUEST_BYTES: usize = 34 * 1024 * 1024;

/// Bounded allowance for re-encoding the accepted request item as a named
/// msgpack map before writing it to the payload store.
pub const MAX_OFFLOADED_SERIALIZATION_OVERHEAD_BYTES: usize = 64 * 1024;

/// Payload-store envelope limit. Modality-specific limits are enforced after
/// the item is fetched: image/document media remain capped at 16 MiB and the
/// audio decoder caps compressed input at 24 MiB.
pub const MAX_OFFLOADED_PAYLOAD_BYTES: usize =
    MAX_EXTRACT_REQUEST_BYTES + MAX_OFFLOADED_SERIALIZATION_OVERHEAD_BYTES;

/// Existing OpenAI-compatibility upper bound, applied defensively per native
/// item for direct queue/local-ingest work that bypasses that request parser.
pub const MAX_IMAGES_PER_ITEM: usize = 16;

/// Mirror of the gateway's per-request chat video cap.
pub const MAX_VIDEOS_PER_ITEM: usize = 1;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum MediaValidationError {
    #[error("item must be a map")]
    ItemNotMap,
    #[error("item contains duplicate '{0}' fields")]
    DuplicateItemField(&'static str),
    #[error("{path} must be a map with a 'data' field")]
    MediaNotMap { path: String },
    #[error("{path} contains duplicate '{field}' fields")]
    DuplicateMediaField { path: String, field: &'static str },
    #[error("{path}.data must be non-empty bytes")]
    InvalidMediaData { path: String },
    #[error("{path}.format must be a string or null")]
    InvalidMediaFormat { path: String },
    #[error("images must be an array or null")]
    ImagesNotArray,
    #[error("too many images ({actual}); maximum is {MAX_IMAGES_PER_ITEM} per item")]
    TooManyImages { actual: usize },
    #[error("too many videos ({actual}); maximum is {MAX_VIDEOS_PER_ITEM} per item")]
    TooManyVideos { actual: usize },
    #[error(
        "media payload is too large ({actual} bytes); maximum is {MAX_MEDIA_BYTES_PER_ITEM} bytes per item"
    )]
    MediaTooLarge { actual: usize },
    #[error("generate must be a map")]
    GenerateNotMap,
    #[error("generate.messages must be an array")]
    GenerateMessagesNotArray,
    #[error("{path} must be a map")]
    GenerateObjectNotMap { path: String },
    #[error("{path} contains duplicate '{field}' fields")]
    DuplicateGenerateField { path: String, field: &'static str },
    #[error("{path} must be an array or null")]
    GenerateImagesNotArray { path: String },
    #[error("{path}.data must be non-empty bytes or a non-empty base64 string")]
    InvalidGenerateImageData { path: String },
    #[error("{path}.data is invalid base64: {message}")]
    InvalidGenerateImageBase64 { path: String, message: String },
}

/// Validate media carried by one native API item.
///
/// Unknown item fields and unknown media format hints are intentionally
/// preserved. Format hints are advisory in the public contract; adapters may
/// sniff formats that the generic sidecar does not understand.
pub fn validate_item_media(item: &Value) -> Result<(), MediaValidationError> {
    let Value::Map(fields) = item else {
        return Err(MediaValidationError::ItemNotMap);
    };

    let images = unique_field(fields, "images")
        .map_err(|_| MediaValidationError::DuplicateItemField("images"))?;
    let document = unique_field(fields, "document")
        .map_err(|_| MediaValidationError::DuplicateItemField("document"))?;

    let mut total_bytes = 0usize;
    if let Some(images) = images.filter(|value| !matches!(value, Value::Nil)) {
        let Value::Array(images) = images else {
            return Err(MediaValidationError::ImagesNotArray);
        };
        if images.len() > MAX_IMAGES_PER_ITEM {
            return Err(MediaValidationError::TooManyImages {
                actual: images.len(),
            });
        }
        for (index, image) in images.iter().enumerate() {
            validate_media_object(image, &format!("images[{index}]"), &mut total_bytes)?;
        }
    }

    if let Some(document) = document.filter(|value| !matches!(value, Value::Nil)) {
        validate_media_object(document, "document", &mut total_bytes)?;
    }

    Ok(())
}

/// Replace rolling-compatible generation image and video base64 strings with
/// msgpack binary after enforcing the public request's aggregate count and
/// byte caps (images and videos share one byte budget).
///
/// Compressed image decode and model-specific transforms remain adapter-owned.
pub fn normalize_generate_media(generate: &mut Value) -> Result<(), MediaValidationError> {
    let Value::Map(generate_fields) = generate else {
        return Err(MediaValidationError::GenerateNotMap);
    };
    let Some(messages) = unique_field_mut(generate_fields, "messages").map_err(|_| {
        MediaValidationError::DuplicateGenerateField {
            path: "generate".to_string(),
            field: "messages",
        }
    })?
    else {
        return Ok(());
    };
    let Value::Array(messages) = messages else {
        return Err(MediaValidationError::GenerateMessagesNotArray);
    };

    let mut image_count = 0usize;
    let mut video_count = 0usize;
    let mut total_bytes = 0usize;
    for (message_index, message) in messages.iter_mut().enumerate() {
        let message_path = format!("generate.messages[{message_index}]");
        let Value::Map(message_fields) = message else {
            return Err(MediaValidationError::GenerateObjectNotMap { path: message_path });
        };
        for field in ["images", "videos"] {
            let Some(media) = unique_field_mut(message_fields, field).map_err(|_| {
                MediaValidationError::DuplicateGenerateField {
                    path: message_path.clone(),
                    field,
                }
            })?
            else {
                continue;
            };
            if matches!(media, Value::Nil) {
                continue;
            }
            let Value::Array(media) = media else {
                return Err(MediaValidationError::GenerateImagesNotArray {
                    path: if field == "images" {
                        message_path.clone()
                    } else {
                        format!("{message_path}.{field}")
                    },
                });
            };
            if field == "images" {
                image_count = image_count.saturating_add(media.len());
                if image_count > MAX_IMAGES_PER_ITEM {
                    return Err(MediaValidationError::TooManyImages {
                        actual: image_count,
                    });
                }
            } else {
                video_count = video_count.saturating_add(media.len());
                if video_count > MAX_VIDEOS_PER_ITEM {
                    return Err(MediaValidationError::TooManyVideos {
                        actual: video_count,
                    });
                }
            }
            for (media_index, item) in media.iter_mut().enumerate() {
                let item_path = format!("{message_path}.{field}[{media_index}]");
                normalize_generate_image(item, &item_path, &mut total_bytes)?;
            }
        }
    }
    Ok(())
}

fn normalize_generate_image(
    image: &mut Value,
    path: &str,
    total_bytes: &mut usize,
) -> Result<(), MediaValidationError> {
    let Value::Map(fields) = image else {
        return Err(MediaValidationError::GenerateObjectNotMap {
            path: path.to_string(),
        });
    };
    let format = unique_field(fields, "format").map_err(|_| {
        MediaValidationError::DuplicateGenerateField {
            path: path.to_string(),
            field: "format",
        }
    })?;
    if let Some(format) = format {
        if !matches!(format, Value::String(_) | Value::Nil) {
            return Err(MediaValidationError::InvalidMediaFormat {
                path: path.to_string(),
            });
        }
    }
    let data = unique_field_mut(fields, "data").map_err(|_| {
        MediaValidationError::DuplicateGenerateField {
            path: path.to_string(),
            field: "data",
        }
    })?;
    let Some(data) = data else {
        return Err(MediaValidationError::InvalidGenerateImageData {
            path: path.to_string(),
        });
    };

    let decoded = match data {
        Value::Binary(bytes) if !bytes.is_empty() => None,
        Value::String(encoded) => {
            let Some(encoded) = encoded.as_str() else {
                return Err(MediaValidationError::InvalidGenerateImageData {
                    path: path.to_string(),
                });
            };
            let encoded = encoded.trim();
            if encoded.is_empty() {
                return Err(MediaValidationError::InvalidGenerateImageData {
                    path: path.to_string(),
                });
            }
            let decoded_upper_bound = encoded.len().saturating_mul(3) / 4;
            if decoded_upper_bound > MAX_MEDIA_BYTES_PER_ITEM {
                return Err(MediaValidationError::MediaTooLarge {
                    actual: decoded_upper_bound,
                });
            }
            Some(
                base64::engine::general_purpose::STANDARD
                    .decode(encoded)
                    .map_err(|error| MediaValidationError::InvalidGenerateImageBase64 {
                        path: path.to_string(),
                        message: error.to_string(),
                    })?,
            )
        }
        _ => {
            return Err(MediaValidationError::InvalidGenerateImageData {
                path: path.to_string(),
            });
        }
    };
    if let Some(decoded) = decoded {
        if decoded.is_empty() {
            return Err(MediaValidationError::InvalidGenerateImageData {
                path: path.to_string(),
            });
        }
        *data = Value::Binary(decoded);
    }
    let Value::Binary(bytes) = data else {
        unreachable!("generation image normalized to msgpack binary");
    };
    *total_bytes = total_bytes.saturating_add(bytes.len());
    if *total_bytes > MAX_MEDIA_BYTES_PER_ITEM {
        return Err(MediaValidationError::MediaTooLarge {
            actual: *total_bytes,
        });
    }
    Ok(())
}

fn validate_media_object(
    value: &Value,
    path: &str,
    total_bytes: &mut usize,
) -> Result<(), MediaValidationError> {
    let Value::Map(fields) = value else {
        return Err(MediaValidationError::MediaNotMap {
            path: path.to_string(),
        });
    };
    let data =
        unique_field(fields, "data").map_err(|_| MediaValidationError::DuplicateMediaField {
            path: path.to_string(),
            field: "data",
        })?;
    let format =
        unique_field(fields, "format").map_err(|_| MediaValidationError::DuplicateMediaField {
            path: path.to_string(),
            field: "format",
        })?;

    let Some(Value::Binary(data)) = data else {
        return Err(MediaValidationError::InvalidMediaData {
            path: path.to_string(),
        });
    };
    if data.is_empty() {
        return Err(MediaValidationError::InvalidMediaData {
            path: path.to_string(),
        });
    }
    if let Some(format) = format {
        if !matches!(format, Value::String(_) | Value::Nil) {
            return Err(MediaValidationError::InvalidMediaFormat {
                path: path.to_string(),
            });
        }
    }

    *total_bytes = total_bytes.saturating_add(data.len());
    if *total_bytes > MAX_MEDIA_BYTES_PER_ITEM {
        return Err(MediaValidationError::MediaTooLarge {
            actual: *total_bytes,
        });
    }
    Ok(())
}

fn unique_field<'a>(fields: &'a [(Value, Value)], expected: &str) -> Result<Option<&'a Value>, ()> {
    let mut found = None;
    for (key, value) in fields {
        if key_as_str(key) != Some(expected) {
            continue;
        }
        if found.is_some() {
            return Err(());
        }
        found = Some(value);
    }
    Ok(found)
}

fn unique_field_mut<'a>(
    fields: &'a mut [(Value, Value)],
    expected: &str,
) -> Result<Option<&'a mut Value>, ()> {
    let mut found = None;
    for (index, (key, _)) in fields.iter().enumerate() {
        if key_as_str(key) != Some(expected) {
            continue;
        }
        if found.replace(index).is_some() {
            return Err(());
        }
    }
    Ok(found.map(|index| &mut fields[index].1))
}

fn key_as_str(value: &Value) -> Option<&str> {
    match value {
        Value::String(value) => value.as_str(),
        Value::Binary(value) => std::str::from_utf8(value).ok(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn media(data: Value, format: Value) -> Value {
        Value::Map(vec![
            (Value::from("data"), data),
            (Value::from("format"), format),
        ])
    }

    fn item(images: Option<Value>, document: Option<Value>) -> Value {
        let mut fields = Vec::new();
        if let Some(images) = images {
            fields.push((Value::from("images"), images));
        }
        if let Some(document) = document {
            fields.push((Value::from("document"), document));
        }
        Value::Map(fields)
    }

    fn generate_with_images(images: Vec<Value>) -> Value {
        Value::Map(vec![(
            Value::from("messages"),
            Value::Array(vec![Value::Map(vec![(
                Value::from("images"),
                Value::Array(images),
            )])]),
        )])
    }

    #[test]
    fn accepts_binary_images_and_document_without_mutation() {
        let value = item(
            Some(Value::Array(vec![media(
                Value::Binary(vec![1, 2, 3]),
                Value::from("jpeg"),
            )])),
            Some(media(
                Value::Binary(b"%PDF".to_vec()),
                Value::from("future-format"),
            )),
        );
        let before = value.clone();
        assert_eq!(validate_item_media(&value), Ok(()));
        assert_eq!(value, before);
    }

    #[test]
    fn accepts_items_without_media_and_null_media_fields() {
        assert_eq!(
            validate_item_media(&Value::Map(vec![(
                Value::from("text"),
                Value::from("hello")
            )])),
            Ok(())
        );
        assert_eq!(
            validate_item_media(&item(Some(Value::Nil), Some(Value::Nil))),
            Ok(())
        );
    }

    #[test]
    fn rejects_non_binary_empty_and_missing_media_data() {
        for invalid in [
            media(Value::from("aGVsbG8="), Value::from("png")),
            media(Value::Binary(Vec::new()), Value::from("png")),
            Value::Map(vec![(Value::from("format"), Value::from("png"))]),
        ] {
            assert!(matches!(
                validate_item_media(&item(Some(Value::Array(vec![invalid])), None)),
                Err(MediaValidationError::InvalidMediaData { .. })
            ));
        }
    }

    #[test]
    fn rejects_non_map_media_and_non_string_format() {
        assert_eq!(
            validate_item_media(&Value::from("not-an-item")),
            Err(MediaValidationError::ItemNotMap)
        );
        assert!(matches!(
            validate_item_media(&item(Some(Value::Array(vec![Value::from("image")])), None)),
            Err(MediaValidationError::MediaNotMap { .. })
        ));
        assert!(matches!(
            validate_item_media(&item(
                Some(Value::Array(vec![media(
                    Value::Binary(vec![1]),
                    Value::from(123)
                )])),
                None,
            )),
            Err(MediaValidationError::InvalidMediaFormat { .. })
        ));
    }

    #[test]
    fn rejects_aggregate_media_bytes_above_ingress_cap() {
        let first = vec![0; MAX_MEDIA_BYTES_PER_ITEM / 2 + 1];
        let second = vec![0; MAX_MEDIA_BYTES_PER_ITEM / 2];
        let err = validate_item_media(&item(
            Some(Value::Array(vec![
                media(Value::Binary(first), Value::Nil),
                media(Value::Binary(second), Value::Nil),
            ])),
            None,
        ))
        .unwrap_err();
        assert_eq!(
            err,
            MediaValidationError::MediaTooLarge {
                actual: MAX_MEDIA_BYTES_PER_ITEM + 1
            }
        );
    }

    #[test]
    fn generation_base64_is_normalized_to_binary() {
        let mut generate =
            generate_with_images(vec![media(Value::from("aGVsbG8="), Value::from("png"))]);
        normalize_generate_media(&mut generate).unwrap();
        let Value::Map(root) = generate else {
            panic!("generate map")
        };
        let Value::Array(messages) = unique_field(&root, "messages").unwrap().unwrap() else {
            panic!("messages")
        };
        let Value::Map(message) = &messages[0] else {
            panic!("message")
        };
        let Value::Array(images) = unique_field(message, "images").unwrap().unwrap() else {
            panic!("images")
        };
        let Value::Map(image) = &images[0] else {
            panic!("image")
        };
        assert_eq!(
            unique_field(image, "data").unwrap(),
            Some(&Value::Binary(b"hello".to_vec()))
        );
    }

    fn generate_with_message(fields: Vec<(&str, Value)>) -> Value {
        Value::Map(vec![(
            Value::from("messages"),
            Value::Array(vec![Value::Map(
                fields
                    .into_iter()
                    .map(|(key, value)| (Value::from(key), value))
                    .collect(),
            )]),
        )])
    }

    #[test]
    fn generation_video_base64_is_normalized_to_binary() {
        let mut generate = generate_with_message(vec![(
            "videos",
            Value::Array(vec![media(Value::from("aGVsbG8="), Value::from("mp4"))]),
        )]);
        normalize_generate_media(&mut generate).unwrap();
        let Value::Map(root) = generate else {
            panic!("generate map")
        };
        let Value::Array(messages) = unique_field(&root, "messages").unwrap().unwrap() else {
            panic!("messages")
        };
        let Value::Map(message) = &messages[0] else {
            panic!("message")
        };
        let Value::Array(videos) = unique_field(message, "videos").unwrap().unwrap() else {
            panic!("videos")
        };
        let Value::Map(video) = &videos[0] else {
            panic!("video")
        };
        assert_eq!(
            unique_field(video, "data").unwrap(),
            Some(&Value::Binary(b"hello".to_vec()))
        );
    }

    #[test]
    fn generation_rejects_second_video_and_shares_byte_budget_with_images() {
        let clip = || media(Value::from("aGVsbG8="), Value::from("mp4"));
        let mut two = generate_with_message(vec![("videos", Value::Array(vec![clip(), clip()]))]);
        assert_eq!(
            normalize_generate_media(&mut two),
            Err(MediaValidationError::TooManyVideos { actual: 2 })
        );

        let half = vec![7u8; MAX_MEDIA_BYTES_PER_ITEM / 2 + 1];
        let mut over = generate_with_message(vec![
            (
                "images",
                Value::Array(vec![media(Value::Binary(half.clone()), Value::Nil)]),
            ),
            (
                "videos",
                Value::Array(vec![media(Value::Binary(half), Value::Nil)]),
            ),
        ]);
        assert!(matches!(
            normalize_generate_media(&mut over),
            Err(MediaValidationError::MediaTooLarge { .. })
        ));

        let mut not_array = generate_with_message(vec![("videos", Value::from("x"))]);
        assert_eq!(
            normalize_generate_media(&mut not_array),
            Err(MediaValidationError::GenerateImagesNotArray {
                path: "generate.messages[0].videos".to_string()
            })
        );
    }

    #[test]
    fn generation_prepared_binary_is_rolling_compatible() {
        let mut generate = generate_with_images(vec![media(
            Value::Binary(b"already prepared".to_vec()),
            Value::Nil,
        )]);
        let before = generate.clone();
        normalize_generate_media(&mut generate).unwrap();
        assert_eq!(generate, before);
    }

    #[test]
    fn generation_media_rejects_invalid_empty_duplicate_and_excess_count() {
        for invalid in [
            Value::from("!!!"),
            Value::from(""),
            Value::Binary(Vec::new()),
        ] {
            let mut generate = generate_with_images(vec![media(invalid, Value::from("png"))]);
            assert!(normalize_generate_media(&mut generate).is_err());
        }

        let mut duplicate = generate_with_images(vec![Value::Map(vec![
            (Value::from("data"), Value::from("YQ==")),
            (Value::from("data"), Value::from("Yg==")),
        ])]);
        assert!(matches!(
            normalize_generate_media(&mut duplicate),
            Err(MediaValidationError::DuplicateGenerateField { .. })
        ));

        let mut too_many = generate_with_images(
            (0..=MAX_IMAGES_PER_ITEM)
                .map(|_| media(Value::Binary(vec![1]), Value::Nil))
                .collect(),
        );
        assert!(matches!(
            normalize_generate_media(&mut too_many),
            Err(MediaValidationError::TooManyImages { .. })
        ));
    }

    #[test]
    fn rejects_image_count_above_existing_limit() {
        let images = (0..=MAX_IMAGES_PER_ITEM)
            .map(|_| media(Value::Binary(vec![1]), Value::Nil))
            .collect();
        assert_eq!(
            validate_item_media(&item(Some(Value::Array(images)), None)),
            Err(MediaValidationError::TooManyImages {
                actual: MAX_IMAGES_PER_ITEM + 1
            })
        );
    }

    #[test]
    fn rejects_ambiguous_duplicate_fields() {
        let duplicate_item = Value::Map(vec![
            (Value::from("images"), Value::Array(Vec::new())),
            (Value::from("images"), Value::Array(Vec::new())),
        ]);
        assert_eq!(
            validate_item_media(&duplicate_item),
            Err(MediaValidationError::DuplicateItemField("images"))
        );

        let duplicate_data = Value::Map(vec![
            (Value::from("data"), Value::Binary(vec![1])),
            (Value::from("data"), Value::Binary(vec![2])),
        ]);
        assert!(matches!(
            validate_item_media(&item(Some(Value::Array(vec![duplicate_data])), None)),
            Err(MediaValidationError::DuplicateMediaField { field: "data", .. })
        ));
    }

    #[test]
    fn max_audio_offload_contract_uses_msgpack_binary_and_fits_envelope() {
        let compressed_bytes = sie_audio_prep::DEFAULT_MAX_COMPRESSED_BYTES;
        let item = Value::Map(vec![(
            Value::from("audio"),
            Value::Map(vec![
                (
                    Value::from("data"),
                    Value::Binary(vec![0x5a; compressed_bytes]),
                ),
                (Value::from("format"), Value::from("wav")),
            ]),
        )]);

        let encoded = rmp_serde::to_vec_named(&item).expect("audio item should encode");
        assert_eq!(MAX_EXTRACT_REQUEST_BYTES, 34 * 1024 * 1024);
        assert_eq!(
            MAX_OFFLOADED_PAYLOAD_BYTES,
            MAX_EXTRACT_REQUEST_BYTES + MAX_OFFLOADED_SERIALIZATION_OVERHEAD_BYTES
        );
        assert!(encoded.len() <= MAX_OFFLOADED_PAYLOAD_BYTES);
        assert!(
            encoded.len() < compressed_bytes + 1024,
            "audio bytes must remain msgpack binary instead of base64"
        );

        drop(item);
        let decoded: Value = rmp_serde::from_slice(&encoded).expect("audio item should decode");
        let Value::Map(item_fields) = decoded else {
            panic!("decoded item must be a map");
        };
        let Some(Value::Map(audio_fields)) =
            unique_field(&item_fields, "audio").expect("audio must be unique")
        else {
            panic!("decoded item.audio must be a map");
        };
        assert!(matches!(
            unique_field(audio_fields, "data").expect("data must be unique"),
            Some(Value::Binary(data)) if data.len() == compressed_bytes
        ));
    }
}
