//! Request selector syntax only. Profile/source authorization remains at ingress.

use serde_json::{Map, Value};

pub(crate) const FIELDS: [&str; 3] = ["lora_adapter", "lora", "lora_id"];

#[derive(Debug)]
pub(crate) struct SelectorError {
    pub field: &'static str,
    pub message: String,
}

/// Retain the supplied spellings so legacy source-only admission cannot be lost.
pub(crate) fn selectors(body: &Map<String, Value>) -> Result<Map<String, Value>, SelectorError> {
    let mut selected = None;
    let mut fields = Map::new();
    for field in FIELDS {
        let Some(value) = body.get(field) else {
            continue;
        };
        if !value.is_null() && !value.as_str().is_some_and(|s| !s.trim().is_empty()) {
            return Err(SelectorError {
                field,
                message: format!("'{field}' must be a non-empty string or null"),
            });
        }
        if selected.is_some_and(|prior| prior != value) {
            return Err(SelectorError {
                field,
                message: "LoRA selector fields must agree, including explicit null values".into(),
            });
        }
        selected = Some(value);
        fields.insert(field.into(), value.clone());
    }
    Ok(fields)
}

pub(crate) fn validate_options(options: Option<&Value>) -> Result<(), String> {
    match options {
        None | Some(Value::Null) => Ok(()),
        Some(Value::Object(options)) => selectors(options)
            .map(|_| ())
            .map_err(|error| error.message),
        _ => Err("'options' must be an object or null".into()),
    }
}

/// Validate before the generic MessagePack-to-JSON bridge can coerce a value.
pub(crate) fn validate_options_rmpv(options: Option<&rmpv::Value>) -> Result<(), String> {
    let fields = match options {
        None | Some(rmpv::Value::Nil) => return Ok(()),
        Some(rmpv::Value::Map(fields)) => fields,
        _ => return Err("'options' must be an object or null".into()),
    };
    let mut selected = Map::new();
    for (key, value) in fields {
        // The generic bridge and ancestor lookup also accept UTF-8 binary keys.
        let key = match key {
            rmpv::Value::String(key) => key.as_str(),
            rmpv::Value::Binary(key) => std::str::from_utf8(key).ok(),
            _ => None,
        };
        let Some(key) = key.filter(|key| FIELDS.contains(key)) else {
            continue;
        };
        let value = match value {
            rmpv::Value::Nil => Value::Null,
            rmpv::Value::String(value) if value.as_str().is_some() => {
                Value::String(value.as_str().unwrap().into())
            }
            _ => return Err(format!("'{key}' must be a non-empty string or null")),
        };
        if selected.insert(key.into(), value).is_some() {
            return Err(format!("duplicate LoRA selector '{key}'"));
        }
    }
    selectors(&selected)
        .map(|_| ())
        .map_err(|error| error.message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn selectors_preserve_absence_null_and_identical_aliases() {
        assert!(selectors(&Map::new()).unwrap().is_empty());
        for value in [Value::Null, json!("legal")] {
            let body = json!({"lora_adapter": value, "lora": value, "lora_id": value});
            assert_eq!(
                selectors(body.as_object().unwrap()).unwrap(),
                *body.as_object().unwrap()
            );
        }
    }

    #[test]
    fn selectors_reject_conflicting_null_and_invalid_values() {
        for body in [
            json!({"lora_adapter": null, "lora": "legal"}),
            json!({"lora_adapter": "legal", "lora_id": "other"}),
            json!({"lora": false}),
            json!({"lora_id": 1}),
            json!({"lora_adapter": " "}),
            json!({"lora_adapter": ["legal"]}),
        ] {
            assert!(selectors(body.as_object().unwrap()).is_err(), "{body}");
        }
    }

    #[test]
    fn msgpack_selector_types_cannot_be_coerced_to_null() {
        for value in [
            rmpv::Value::F64(f64::NAN),
            rmpv::Value::Binary(vec![1]),
            rmpv::Value::Ext(1, vec![1]),
            rmpv::Value::Boolean(false),
        ] {
            for key in FIELDS {
                for key in [key.into(), rmpv::Value::Binary(key.as_bytes().to_vec())] {
                    let options = rmpv::Value::Map(vec![(key, value.clone())]);
                    assert!(validate_options_rmpv(Some(&options)).is_err());
                }
            }
        }
    }

    #[test]
    fn msgpack_binary_lora_keys_preserve_selection_and_reject_duplicates() {
        for key in FIELDS {
            let binary_key = rmpv::Value::Binary(key.as_bytes().to_vec());
            for value in [rmpv::Value::Nil, "legal".into()] {
                let options = rmpv::Value::Map(vec![(binary_key.clone(), value)]);
                assert!(validate_options_rmpv(Some(&options)).is_ok());
            }
            let empty = rmpv::Value::Map(vec![(binary_key.clone(), " ".into())]);
            assert!(validate_options_rmpv(Some(&empty)).is_err());
            for duplicate_first in [false, true] {
                let mut fields = vec![
                    (key.into(), "legal".into()),
                    (binary_key.clone(), rmpv::Value::Nil),
                ];
                if duplicate_first {
                    fields.reverse();
                }
                let error = validate_options_rmpv(Some(&rmpv::Value::Map(fields))).unwrap_err();
                assert!(error.contains("duplicate LoRA selector"));
            }
        }
        let conflict = rmpv::Value::Map(vec![
            (
                rmpv::Value::Binary(b"lora_adapter".to_vec()),
                rmpv::Value::Nil,
            ),
            (rmpv::Value::Binary(b"lora".to_vec()), "legal".into()),
        ]);
        assert!(validate_options_rmpv(Some(&conflict)).is_err());
    }
}
