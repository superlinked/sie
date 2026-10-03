//! Sorted-key JSON for hashes and generated artifacts.
//!
//! The gateway builds `serde_json` with `preserve_order`, so a
//! `serde_json::Value` object serialises its keys in insertion order. That
//! is what request forwarding needs: a caller's `json_schema` property
//! order decides the order a constrained decoder emits fields in, so it
//! must reach the worker untouched.
//!
//! Anything whose bytes are compared across processes or languages (a
//! content hash matched against another service, a checked-in generated
//! file) must not depend on insertion order. Those call sites run their
//! value through [`sorted`] before serialising, which reproduces the
//! sorted-key output of Python's `json.dumps(..., sort_keys=True)`: object
//! keys ordered by Unicode code point, recursively, arrays left in place.

use serde_json::{Map, Value};

/// Return `value` with every object's keys sorted, recursively.
///
/// Rust `String` ordering is byte-wise UTF-8, which orders the same as
/// Unicode code points, so the result matches Python's `sort_keys=True`.
pub fn sorted(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_unstable_by_key(|(key, _)| *key);
            let mut out = Map::with_capacity(entries.len());
            for (key, child) in entries {
                out.insert(key.clone(), sorted(child));
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.iter().map(sorted).collect()),
        other => other.clone(),
    }
}

/// Serialise `value` as compact JSON with every object's keys sorted.
pub fn to_sorted_string(value: &Value) -> String {
    serde_json::to_string(&sorted(value)).expect("serde_json::Value always serialises")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorted_orders_keys_recursively_and_keeps_array_order() {
        let value: Value = serde_json::from_str(
            r#"{"zeta":{"b":1,"a":[{"y":1,"x":2},3]},"alpha":null,"Mid":true}"#,
        )
        .unwrap();
        assert_eq!(
            to_sorted_string(&value),
            r#"{"Mid":true,"alpha":null,"zeta":{"a":[{"x":2,"y":1},3],"b":1}}"#
        );
    }

    #[test]
    fn sorted_output_does_not_depend_on_input_order() {
        let a: Value = serde_json::from_str(r#"{"b":{"d":1,"c":2},"a":0}"#).unwrap();
        let b: Value = serde_json::from_str(r#"{"a":0,"b":{"c":2,"d":1}}"#).unwrap();
        assert_ne!(
            serde_json::to_string(&a).unwrap(),
            serde_json::to_string(&b).unwrap(),
            "preserve_order must keep the caller's key order in plain serialisation"
        );
        assert_eq!(to_sorted_string(&a), to_sorted_string(&b));
    }
}
