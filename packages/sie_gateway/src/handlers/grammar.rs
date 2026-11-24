//! Shared grammar (structured-output) parser and safety caps.
//!
//! Both the SIE-native ``/v1/generate/{model}`` JSON parser and the
//! OpenAI ``/v1/chat/completions`` ``response_format`` translator funnel
//! through :func:`parse_grammar`. The parser enforces:
//!
//! * Payload size cap (64 KiB)
//! * JSON Schema nesting-depth cap (16)
//! * Raw traversal-depth cap (128)
//! * Regex length cap (4 KiB)
//! * Internal JSON Schema ``$ref`` dereferencing (``#/...`` only)
//! * JSON Schema reject-list (``$dynamicRef``, ``if/then/else``,
//!   ``unevaluatedProperties``, ``dependentSchemas``)
//! * Keyword value shapes every JSON Schema draft refuses (an unknown
//!   ``type`` name, ``properties`` that is not an object, ``required``
//!   that is not an array of strings, a string where a schema belongs, …)
//! * Mutual exclusivity of ``json_schema`` and ``regex``
//! * ``strict: true`` only on ``json_schema`` and ``regex`` grammars, whose
//!   finished output the worker verifies
//!
//! All failures return a 400 :class:`Response` carrying the OpenAI
//! error envelope with ``code`` (``grammar_invalid`` |
//! ``unsupported_field``) and ``param`` naming the offending key path.
//! The worker is downstream and assumes the gateway has already
//! filtered — it does not re-check these caps.
//!
//! ``$ref`` support is deliberately bounded to internal JSON pointers
//! (``#/...``). External documents, unresolved pointers, and recursive
//! reference cycles are rejected before the worker sees the schema.
//!
//! Tool parameter schemas go through :func:`validate_tool_parameters`
//! instead. They are never compiled into a grammar, so they are checked as
//! sent, without ``$ref`` expansion.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};

use crate::http_error::{json_openai_error, openai_code as oai_code, openai_type as oai_type};
use crate::queue::publisher::GrammarSpec;

/// Helper that maps a :class:`GrammarSpec` variant back to the string
/// the model's ``capabilities.grammar`` list uses
/// (``"json_schema"`` | ``"regex"`` | ``"ebnf"``).
pub fn grammar_kind_label(g: &GrammarSpec) -> &'static str {
    match g {
        GrammarSpec::JsonSchema { .. } => "json_schema",
        GrammarSpec::Regex { .. } => "regex",
        GrammarSpec::Ebnf { .. } => "ebnf",
    }
}

/// Reject the request when the model's
/// ``tasks.generate.capabilities.grammar`` list does not advertise the
/// requested ``grammar.kind``. ``capabilities`` is the YAML-derived
/// list as exposed by :class:`ModelInfoExtras`. A ``None`` capabilities
/// list (model has no ``generate`` task at all) also rejects — a
/// non-generation model cannot accept ``grammar`` regardless. Returns
/// ``Ok(())`` when the request is permitted.
#[allow(clippy::result_large_err)]
pub fn check_capability(
    grammar: &GrammarSpec,
    capabilities: Option<&[String]>,
    model: &str,
) -> Result<(), Response> {
    let kind = grammar_kind_label(grammar);
    let allowed = capabilities.is_some_and(|caps| caps.iter().any(|c| c == kind));
    if allowed {
        return Ok(());
    }
    let param = format!("grammar.{kind}");
    let message = if capabilities.is_none() {
        format!("Model '{model}' does not support grammar (no generate task)")
    } else {
        format!("Model '{model}' does not declare '{kind}' grammar support")
    };
    Err((
        StatusCode::BAD_REQUEST,
        Json(json_openai_error(
            message,
            oai_type::INVALID_REQUEST,
            Some(&param),
            oai_code::UNSUPPORTED_FIELD,
        )),
    )
        .into_response())
}

/// Maximum size of the raw ``grammar`` object after JSON serialisation,
/// in bytes. 64 KiB comfortably fits the typical extraction schemas
/// while keeping the worker compile budget bounded.
pub const MAX_GRAMMAR_BYTES: usize = 64 * 1024;

/// Maximum size of a tool's parameter schema as sent, in bytes. The same
/// budget as a grammar, but measured without ``$ref`` expansion; see
/// :func:`validate_tool_parameters`.
pub const MAX_TOOL_SCHEMA_BYTES: usize = MAX_GRAMMAR_BYTES;

/// Maximum JSON Schema nesting depth. Counted by recursive walks over
/// ``properties`` / ``items`` / ``oneOf`` / ``anyOf`` / ``allOf`` /
/// ``additionalProperties``. Pathological schemas (think
/// ``{"items":{"items":{"items":...}}}``) trigger this before Outlines
/// gets a chance to OOM on compile.
pub const MAX_SCHEMA_DEPTH: usize = 16;

/// Maximum raw tree / reference-expansion depth visited by either schema
/// pass. This is deliberately independent of :const:`MAX_SCHEMA_DEPTH`:
/// unknown annotation keys do not increase semantic schema depth, but they
/// still consume a Rust recursion frame when a programmatically constructed
/// or alternate-ingress :class:`Value` reaches this validator.
pub const MAX_SCHEMA_TRAVERSAL_DEPTH: usize = 128;

/// Maximum total node count visited during the schema walk. Depth alone
/// doesn't stop a *wide* schema (one shallow object with hundreds of
/// thousands of trivial properties), where every key still pays for a
/// `format!` allocation and a recursion frame. 16 384 is far above any
/// legitimate schema while still bounding the walker's CPU/allocations
/// to single-digit milliseconds in the worst case.
pub const MAX_SCHEMA_NODES: usize = 16 * 1024;

/// Maximum regex length, in characters. Long regexes drive the
/// Outlines compile time non-linearly; 4 KiB is generous for legitimate
/// use cases (license keys, product codes, etc.).
pub const MAX_REGEX_LEN: usize = 4 * 1024;

/// Maximum EBNF/CFG grammar source length, in characters. Outlines'
/// EBNF compiler is exponential in the worst case; we cap the source
/// well below :const:`MAX_GRAMMAR_BYTES` so the overall payload cap
/// still leaves room for label/strict siblings without surprising the
/// caller with a payload-size rejection on a grammar that fits.
pub const MAX_EBNF_LEN: usize = 8 * 1024;

/// JSON Schema keywords rejected at the gateway. These are features
/// Outlines either does not implement or implements at prohibitive
/// compile cost. Each rejection names the keyword in ``param`` so
/// callers can fix their schema without trial-and-error.
const UNSUPPORTED_KEYWORDS: &[&str] = &[
    "$dynamicRef",
    "if",
    "then",
    "else",
    "unevaluatedProperties",
    "dependentSchemas",
];

/// Result of :func:`parse_grammar`. ``Err`` carries an already-built
/// 400 response so the caller can return it directly (mirrors
/// :class:`ChatParamsResult` in ``proxy.rs``).
pub enum GrammarParseResult {
    Ok(GrammarSpec),
    Err(Response),
}

fn bad_request(message: String, param: &str, code: &'static str) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(json_openai_error(
            message,
            oai_type::INVALID_REQUEST,
            Some(param),
            code,
        )),
    )
        .into_response()
}

/// Why a JSON Schema was refused: the 400's ``message``, the ``param``
/// path it names, and its error ``code``.
#[derive(Debug)]
pub struct SchemaRejection {
    pub message: String,
    pub param: String,
    pub code: &'static str,
}

impl SchemaRejection {
    fn new(message: String, param: &str, code: &'static str) -> Self {
        Self {
            message,
            param: param.to_string(),
            code,
        }
    }

    pub fn into_response(self) -> Response {
        bad_request(self.message, &self.param, self.code)
    }
}

/// Parse the ``grammar`` object as it appears under
/// ``/v1/generate/{model}`` request bodies. Caller is responsible for
/// the capability gate (model config ``tasks.generate.capabilities.grammar``
/// list); this function only enforces the wire-shape contract.
///
/// Wire shape (mutually-exclusive variants under ``grammar:``):
///
/// ```json
/// { "json_schema": {"type": "object", ...} }
/// // or
/// { "regex": "[A-Z]{3}-\\d{4}" }
/// // or
/// { "ebnf": "root ::= \"hello\" | \"goodbye\"" }
/// ```
///
/// Optional sibling keys ``label`` and ``strict`` are surfaced to the
/// worker via the resulting :class:`GrammarSpec`. They never affect the
/// cache key — see :func:`sie_server.types.grammar.hash_grammar`.
pub fn parse_grammar(v: &Value) -> GrammarParseResult {
    parse_grammar_at(v, "grammar.json_schema")
}

/// :func:`parse_grammar` with the ``param`` path that schema rejections
/// name for the ``json_schema`` variant. Callers that translate another
/// request field into a grammar pass that field's path, so a refused
/// schema is reported where the client sent it.
pub fn parse_grammar_at(v: &Value, schema_param: &str) -> GrammarParseResult {
    let Some(obj) = v.as_object() else {
        return GrammarParseResult::Err(bad_request(
            "'grammar' must be a JSON object".to_string(),
            "grammar",
            oai_code::INVALID_REQUEST,
        ));
    };

    // §2.5 step 1: payload size cap. Run before any recursion so a
    // billion-key schema cannot exhaust the walker's stack before the
    // size check fires.
    //
    // ``serde_json::to_vec`` is the cheapest way to get a byte count
    // for a tree we already hold; we deliberately don't compare against
    // the request body length because the gateway's outer
    // ``MAX_PROXY_BODY`` covers that and `grammar:` is one field of
    // many.
    let serialized_len = serde_json::to_vec(v).map(|b| b.len()).unwrap_or(0);
    if serialized_len > MAX_GRAMMAR_BYTES {
        return GrammarParseResult::Err(bad_request(
            format!(
                "grammar payload {serialized_len} bytes exceeds limit ({MAX_GRAMMAR_BYTES} bytes)"
            ),
            "grammar",
            oai_code::INVALID_REQUEST,
        ));
    }

    let has_schema = obj.contains_key("json_schema");
    let has_regex = obj.contains_key("regex");
    let has_ebnf = obj.contains_key("ebnf");
    let variants_present = [has_schema, has_regex, has_ebnf]
        .iter()
        .filter(|p| **p)
        .count();
    if variants_present > 1 {
        return GrammarParseResult::Err(bad_request(
            "'grammar.json_schema', 'grammar.regex' and 'grammar.ebnf' are mutually exclusive"
                .to_string(),
            "grammar",
            oai_code::INVALID_REQUEST,
        ));
    }
    if variants_present == 0 {
        return GrammarParseResult::Err(bad_request(
            "'grammar' must contain exactly one of 'json_schema', 'regex' or 'ebnf'".to_string(),
            "grammar",
            oai_code::INVALID_REQUEST,
        ));
    }

    let label = obj.get("label").and_then(|v| v.as_str()).map(String::from);
    let strict = obj.get("strict").and_then(|v| v.as_bool());

    if has_schema {
        let schema = obj.get("json_schema").expect("checked above");
        let resolved_schema = match dereference_schema_refs(schema, schema_param) {
            Ok(schema) => schema,
            Err(rejection) => return GrammarParseResult::Err(rejection.into_response()),
        };
        let serialized_len =
            json_schema_grammar_len(&resolved_schema, label.as_deref(), strict).unwrap_or(0);
        if serialized_len > MAX_GRAMMAR_BYTES {
            return GrammarParseResult::Err(bad_request(
                format!(
                    "grammar payload {serialized_len} bytes exceeds limit ({MAX_GRAMMAR_BYTES} bytes)"
                ),
                "grammar",
                oai_code::INVALID_REQUEST,
            ));
        }
        if let Err(rejection) = walk_schema(&resolved_schema, schema_param, None) {
            return GrammarParseResult::Err(rejection.into_response());
        }
        GrammarParseResult::Ok(GrammarSpec::JsonSchema {
            value: resolved_schema,
            label,
            strict,
        })
    } else if has_regex {
        let regex_val = obj.get("regex").expect("checked above");
        let Some(regex) = regex_val.as_str() else {
            return GrammarParseResult::Err(bad_request(
                "'grammar.regex' must be a string".to_string(),
                "grammar.regex",
                oai_code::INVALID_REQUEST,
            ));
        };
        if regex.len() > MAX_REGEX_LEN {
            return GrammarParseResult::Err(bad_request(
                format!(
                    "regex length {} exceeds limit ({MAX_REGEX_LEN})",
                    regex.len()
                ),
                "grammar.regex",
                oai_code::INVALID_REQUEST,
            ));
        }
        GrammarParseResult::Ok(GrammarSpec::Regex {
            value: regex.to_string(),
            label,
            strict,
        })
    } else {
        // ``ebnf`` branch — string source; no further structural walk
        // (the gateway does not parse EBNF; Outlines/XGrammar is the
        // authority). MAX_GRAMMAR_BYTES at the envelope level plus
        // MAX_EBNF_LEN at the source level bound compile cost.
        let ebnf_val = obj.get("ebnf").expect("checked above");
        let Some(ebnf) = ebnf_val.as_str() else {
            return GrammarParseResult::Err(bad_request(
                "'grammar.ebnf' must be a string".to_string(),
                "grammar.ebnf",
                oai_code::INVALID_REQUEST,
            ));
        };
        if ebnf.len() > MAX_EBNF_LEN {
            return GrammarParseResult::Err(bad_request(
                format!("ebnf length {} exceeds limit ({MAX_EBNF_LEN})", ebnf.len()),
                "grammar.ebnf",
                oai_code::INVALID_REQUEST,
            ));
        }
        if strict == Some(true) {
            return GrammarParseResult::Err(bad_request(
                "'grammar.strict' is not supported for EBNF grammars".to_string(),
                "grammar.strict",
                oai_code::UNSUPPORTED_FIELD,
            ));
        }
        GrammarParseResult::Ok(GrammarSpec::Ebnf {
            value: ebnf.to_string(),
            label,
            strict,
        })
    }
}

fn json_schema_grammar_len(
    schema: &Value,
    label: Option<&str>,
    strict: Option<bool>,
) -> Result<usize, serde_json::Error> {
    let mut obj = serde_json::Map::new();
    obj.insert("json_schema".to_string(), schema.clone());
    if let Some(label) = label {
        obj.insert("label".to_string(), Value::String(label.to_string()));
    }
    if let Some(strict) = strict {
        obj.insert("strict".to_string(), Value::Bool(strict));
    }
    serde_json::to_vec(&Value::Object(obj)).map(|b| b.len())
}

/// Check a tool's ``parameters`` (or ``input_schema``) before the request
/// is forwarded. ``path`` is the field's request path, used in ``param``.
///
/// The worker renders tool schemas into the prompt and reads them to parse
/// the model's calls; it never compiles them into a grammar (a forced
/// ``tool_choice`` constrains only the tool name), and it follows ``$ref``
/// lazily with its own bound. So the schema is checked as sent, without
/// ``$ref`` expansion: its serialized size, depth and node count, the
/// unsupported keywords, and that every ``$ref`` names a definition inside
/// it. A schema that reuses definitions is measured as written, and a
/// recursive definition is allowed.
pub fn validate_tool_parameters(schema: &Value, path: &str) -> Result<(), SchemaRejection> {
    let serialized_len = serde_json::to_vec(schema).map(|b| b.len()).unwrap_or(0);
    if serialized_len > MAX_TOOL_SCHEMA_BYTES {
        return Err(SchemaRejection::new(
            format!(
                "JSON Schema {serialized_len} bytes exceeds limit ({MAX_TOOL_SCHEMA_BYTES} bytes)"
            ),
            path,
            oai_code::INVALID_REQUEST,
        ));
    }
    walk_schema(schema, path, Some(schema))
}

/// Resolve an internal ``$ref`` against ``root``, returning the ref
/// string (``#`` plus its JSON pointer) and its target. ``ref_param`` is
/// the ``param`` path of the ``$ref`` key itself.
fn resolve_internal_ref<'r, 'v>(
    root: &'r Value,
    ref_value: &'v Value,
    ref_param: &str,
) -> Result<(&'v str, &'r Value), SchemaRejection> {
    let Some(ref_str) = ref_value.as_str() else {
        return Err(SchemaRejection::new(
            "'$ref' must be a string".to_string(),
            ref_param,
            oai_code::INVALID_REQUEST,
        ));
    };
    let Some(pointer) = ref_str.strip_prefix('#') else {
        return Err(SchemaRejection::new(
            "external '$ref' is not supported".to_string(),
            ref_param,
            oai_code::UNSUPPORTED_FIELD,
        ));
    };
    if !pointer.is_empty() && !pointer.starts_with('/') {
        return Err(SchemaRejection::new(
            "only internal JSON-pointer '$ref' values are supported".to_string(),
            ref_param,
            oai_code::UNSUPPORTED_FIELD,
        ));
    }
    let Some(target) = root.pointer(pointer) else {
        return Err(SchemaRejection::new(
            format!("unresolved internal '$ref' {ref_str:?}"),
            ref_param,
            oai_code::INVALID_REQUEST,
        ));
    };
    Ok((ref_str, target))
}

fn dereference_schema_refs(schema: &Value, path: &str) -> Result<Value, SchemaRejection> {
    let mut visited: usize = 0;
    let mut stack: Vec<String> = Vec::new();
    dereference_schema_refs_inner(
        schema,
        schema,
        path,
        &mut visited,
        &mut stack,
        SchemaContext::Schema,
        0,
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SchemaContext {
    Schema,
    SchemaMap,
    SchemaArray,
    Other,
}

fn dereference_schema_refs_inner(
    root: &Value,
    v: &Value,
    path: &str,
    visited: &mut usize,
    stack: &mut Vec<String>,
    context: SchemaContext,
    traversal_depth: usize,
) -> Result<Value, SchemaRejection> {
    *visited = visited.saturating_add(1);
    if *visited > MAX_SCHEMA_NODES {
        return Err(SchemaRejection::new(
            format!("JSON Schema node count exceeds limit ({MAX_SCHEMA_NODES})"),
            path,
            oai_code::INVALID_REQUEST,
        ));
    }
    if traversal_depth > MAX_SCHEMA_TRAVERSAL_DEPTH {
        return Err(SchemaRejection::new(
            format!("JSON Schema traversal depth exceeds limit ({MAX_SCHEMA_TRAVERSAL_DEPTH})"),
            path,
            oai_code::INVALID_REQUEST,
        ));
    }

    match v {
        Value::Object(map) => {
            // Definitions are dropped below once their references are
            // inlined, so their shape is checked here, before they go.
            if context == SchemaContext::Schema {
                check_definitions_shape(map, path)?;
            }
            if context == SchemaContext::Schema && map.contains_key("$ref") {
                let ref_param = format!("{path}.$ref");
                let (ref_str, target) = resolve_internal_ref(
                    root,
                    map.get("$ref").expect("checked above"),
                    &ref_param,
                )?;
                // A ref on the stack resolved when it was pushed, so
                // checking the cycle after resolution reports the same error.
                if stack.iter().any(|r| r == ref_str) {
                    return Err(SchemaRejection::new(
                        format!("recursive '$ref' cycle detected at {ref_str:?}"),
                        &ref_param,
                        oai_code::INVALID_REQUEST,
                    ));
                }

                stack.push(ref_str.to_string());
                let resolved = dereference_schema_refs_inner(
                    root,
                    target,
                    path,
                    visited,
                    stack,
                    SchemaContext::Schema,
                    traversal_depth + 1,
                )?;
                stack.pop();

                let mut siblings = serde_json::Map::new();
                for (k, child) in map {
                    if k == "$ref" || k == "$defs" || k == "definitions" {
                        continue;
                    }
                    let child_path = format!("{path}.{k}");
                    let child_context = schema_child_context(context, k);
                    siblings.insert(
                        k.clone(),
                        dereference_schema_refs_inner(
                            root,
                            child,
                            &child_path,
                            visited,
                            stack,
                            child_context,
                            traversal_depth + 1,
                        )?,
                    );
                }

                if siblings.is_empty() {
                    return Ok(resolved);
                }

                // JSON Schema evaluates $ref siblings alongside the
                // referenced schema. Preserve that intersection instead
                // of overwriting same-named object keywords such as
                // properties, required, or allOf.
                Ok(json!({"allOf": [resolved, Value::Object(siblings)]}))
            } else {
                let mut out = serde_json::Map::new();
                for (k, child) in map {
                    if context == SchemaContext::Schema && (k == "$defs" || k == "definitions") {
                        continue;
                    }
                    let child_path = format!("{path}.{k}");
                    let child_context = schema_child_context(context, k);
                    out.insert(
                        k.clone(),
                        dereference_schema_refs_inner(
                            root,
                            child,
                            &child_path,
                            visited,
                            stack,
                            child_context,
                            traversal_depth + 1,
                        )?,
                    );
                }
                Ok(Value::Object(out))
            }
        }
        Value::Array(arr) => {
            let mut out = Vec::with_capacity(arr.len());
            for (i, child) in arr.iter().enumerate() {
                let child_path = format!("{path}[{i}]");
                let child_context =
                    if matches!(context, SchemaContext::Schema | SchemaContext::SchemaArray) {
                        SchemaContext::Schema
                    } else {
                        SchemaContext::Other
                    };
                out.push(dereference_schema_refs_inner(
                    root,
                    child,
                    &child_path,
                    visited,
                    stack,
                    child_context,
                    traversal_depth + 1,
                )?);
            }
            Ok(Value::Array(out))
        }
        _ => Ok(v.clone()),
    }
}

/// ``$defs`` / ``definitions`` must map names to schemas.
fn check_definitions_shape(
    map: &serde_json::Map<String, Value>,
    path: &str,
) -> Result<(), SchemaRejection> {
    for kw in ["$defs", "definitions"] {
        if let Some(v) = map.get(kw).filter(|v| !v.is_null()) {
            if !v.is_object() {
                return Err(invalid_schema(
                    format!(
                        "'{kw}' must be an object mapping names to schemas, got {}",
                        json_kind(v)
                    ),
                    &format!("{path}.{kw}"),
                ));
            }
        }
    }
    Ok(())
}

/// An array where a single schema belongs is never a schema. The caller
/// excludes the tuple form of ``items`` and the members of definition maps.
fn reject_array_schema(v: &Value, path: &str) -> Result<(), SchemaRejection> {
    if v.is_array() {
        return Err(invalid_schema(
            "a JSON Schema must be an object or a boolean".to_string(),
            path,
        ));
    }
    Ok(())
}

fn schema_child_context(parent: SchemaContext, key: &str) -> SchemaContext {
    match parent {
        SchemaContext::Schema => match key {
            "properties" | "patternProperties" | "$defs" | "definitions" | "dependentSchemas" => {
                SchemaContext::SchemaMap
            }
            "oneOf" | "anyOf" | "allOf" | "prefixItems" => SchemaContext::SchemaArray,
            "items"
            | "additionalProperties"
            | "contains"
            | "propertyNames"
            | "not"
            | "if"
            | "then"
            | "else" => SchemaContext::Schema,
            _ => SchemaContext::Other,
        },
        SchemaContext::SchemaMap => SchemaContext::Schema,
        SchemaContext::SchemaArray | SchemaContext::Other => SchemaContext::Other,
    }
}

/// Recursive walk over a JSON-Schema-shaped value. Enforces depth and
/// rejects the unsupported keywords listed in
/// :const:`UNSUPPORTED_KEYWORDS`.
///
/// ``path`` is the dotted accessor for whatever produced 400s name in
/// ``param``. Array elements append ``[N]``; object members append
/// ``.<key>``.
///
/// With ``refs_root``, every ``$ref`` must resolve inside it (see
/// :func:`resolve_internal_ref`); refs are checked, not followed.
fn walk_schema(v: &Value, path: &str, refs_root: Option<&Value>) -> Result<(), SchemaRejection> {
    reject_array_schema(v, path)?;
    let mut visited: usize = 0;
    walk_schema_inner(
        v,
        path,
        0,
        &mut visited,
        SchemaContext::Schema,
        0,
        refs_root,
    )
}

fn walk_schema_inner(
    v: &Value,
    path: &str,
    depth: usize,
    visited: &mut usize,
    context: SchemaContext,
    traversal_depth: usize,
    refs_root: Option<&Value>,
) -> Result<(), SchemaRejection> {
    *visited = visited.saturating_add(1);
    if *visited > MAX_SCHEMA_NODES {
        return Err(SchemaRejection::new(
            format!("JSON Schema node count exceeds limit ({MAX_SCHEMA_NODES})"),
            path,
            oai_code::INVALID_REQUEST,
        ));
    }
    if traversal_depth > MAX_SCHEMA_TRAVERSAL_DEPTH {
        return Err(SchemaRejection::new(
            format!("JSON Schema traversal depth exceeds limit ({MAX_SCHEMA_TRAVERSAL_DEPTH})"),
            path,
            oai_code::INVALID_REQUEST,
        ));
    }
    if depth > MAX_SCHEMA_DEPTH {
        return Err(SchemaRejection::new(
            format!("JSON Schema depth exceeds limit ({MAX_SCHEMA_DEPTH})"),
            path,
            oai_code::INVALID_REQUEST,
        ));
    }

    match v {
        Value::Object(map) => {
            // Only schema objects own JSON Schema keywords. Keys below
            // ``properties`` / ``patternProperties`` / definitions are user
            // names, and annotation/custom-keyword payloads are arbitrary
            // JSON; neither may be mistaken for unsupported keywords.
            if context == SchemaContext::Schema {
                for &kw in UNSUPPORTED_KEYWORDS {
                    if map.contains_key(kw) {
                        let param = format!("{path}.{kw}");
                        let message = format!("JSON Schema keyword '{kw}' is not supported");
                        return Err(SchemaRejection::new(
                            message,
                            &param,
                            oai_code::UNSUPPORTED_FIELD,
                        ));
                    }
                }
                if let (Some(root), Some(ref_value)) = (refs_root, map.get("$ref")) {
                    resolve_internal_ref(root, ref_value, &format!("{path}.$ref"))?;
                }
                check_schema_keyword_shapes(map, path)?;
            }
            for (k, child) in map {
                let child_path = format!("{path}.{k}");
                let child_depth = if context == SchemaContext::Schema && is_schema_nesting_key(k) {
                    depth + 1
                } else {
                    depth
                };
                let child_context = schema_child_context(context, k);
                if context == SchemaContext::Schema {
                    // ``items`` also takes the tuple form, an array of schemas.
                    if child_context == SchemaContext::Schema && k != "items" {
                        reject_array_schema(child, &child_path)?;
                    }
                    // Members of a property map are schemas. Definition maps
                    // are left alone: a ``$ref`` may point into an array held
                    // there, and that is reported as a reference error.
                    if matches!(
                        k.as_str(),
                        "properties" | "patternProperties" | "dependentSchemas"
                    ) {
                        if let Some(members) = child.as_object() {
                            for (name, member) in members {
                                reject_array_schema(member, &format!("{child_path}.{name}"))?;
                            }
                        }
                    }
                }
                walk_schema_inner(
                    child,
                    &child_path,
                    child_depth,
                    visited,
                    child_context,
                    traversal_depth + 1,
                    refs_root,
                )?;
            }
        }
        Value::Array(arr) => {
            let child_context =
                if matches!(context, SchemaContext::Schema | SchemaContext::SchemaArray) {
                    SchemaContext::Schema
                } else {
                    SchemaContext::Other
                };
            for (i, child) in arr.iter().enumerate() {
                let child_path = format!("{path}[{i}]");
                if child_context == SchemaContext::Schema {
                    reject_array_schema(child, &child_path)?;
                }
                walk_schema_inner(
                    child,
                    &child_path,
                    depth,
                    visited,
                    child_context,
                    traversal_depth + 1,
                    refs_root,
                )?;
            }
        }
        // A schema is an object or a boolean. A string or number where a
        // schema belongs (``"properties": {"a": "string"}``) is never a
        // schema. ``null`` is left alone: it is read as an absent keyword.
        Value::String(_) | Value::Number(_) if context == SchemaContext::Schema => {
            return Err(invalid_schema(
                "a JSON Schema must be an object or a boolean".to_string(),
                path,
            ));
        }
        // Other scalars cannot host unsupported keywords and do not
        // contribute to depth.
        _ => {}
    }

    Ok(())
}

/// The type names JSON Schema defines for ``type``.
const SCHEMA_TYPE_NAMES: &[&str] = &[
    "array", "boolean", "integer", "null", "number", "object", "string",
];

/// Keywords whose value is a map from names to schemas.
const SCHEMA_MAP_KEYWORDS: &[&str] = &["properties", "patternProperties", "$defs", "definitions"];

/// Keywords whose value is a non-empty array of schemas.
const SCHEMA_ARRAY_KEYWORDS: &[&str] = &["allOf", "anyOf", "oneOf"];

/// Keywords whose value is a non-negative integer.
const NON_NEGATIVE_INTEGER_KEYWORDS: &[&str] = &[
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "minProperties",
    "maxProperties",
    "minContains",
    "maxContains",
];

/// Keywords whose value is a number.
const NUMBER_KEYWORDS: &[&str] = &["minimum", "maximum", "multipleOf"];

fn invalid_schema(message: String, param: &str) -> SchemaRejection {
    SchemaRejection::new(message, param, oai_code::INVALID_REQUEST)
}

fn json_kind(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

fn check_type_name(v: &Value, param: &str) -> Result<(), SchemaRejection> {
    match v.as_str() {
        Some(name) if SCHEMA_TYPE_NAMES.contains(&name) => Ok(()),
        Some(name) => Err(invalid_schema(
            format!(
                "'type' {name:?} is not a JSON Schema type; expected one of {}",
                SCHEMA_TYPE_NAMES.join(", ")
            ),
            param,
        )),
        None => Err(invalid_schema(
            format!("'type' entries must be strings, got {}", json_kind(v)),
            param,
        )),
    }
}

/// Refuse keyword values that no JSON Schema draft allows, one schema
/// object at a time (the caller walks the children). Only shapes every
/// draft's metaschema refuses are checked; unknown keywords, annotations
/// and ``null`` values pass.
fn check_schema_keyword_shapes(
    map: &serde_json::Map<String, Value>,
    path: &str,
) -> Result<(), SchemaRejection> {
    let present = |key: &str| map.get(key).filter(|v| !v.is_null());

    if let Some(ty) = present("type") {
        let param = format!("{path}.type");
        match ty {
            Value::String(_) => check_type_name(ty, &param)?,
            Value::Array(names) => {
                if names.is_empty() {
                    return Err(invalid_schema(
                        "'type' must not be an empty array".to_string(),
                        &param,
                    ));
                }
                for (i, name) in names.iter().enumerate() {
                    check_type_name(name, &format!("{param}[{i}]"))?;
                }
            }
            other => {
                return Err(invalid_schema(
                    format!(
                        "'type' must be a string or an array of strings, got {}",
                        json_kind(other)
                    ),
                    &param,
                ));
            }
        }
    }

    for &kw in SCHEMA_MAP_KEYWORDS {
        if let Some(v) = present(kw) {
            if !v.is_object() {
                return Err(invalid_schema(
                    format!(
                        "'{kw}' must be an object mapping names to schemas, got {}",
                        json_kind(v)
                    ),
                    &format!("{path}.{kw}"),
                ));
            }
        }
    }

    if let Some(v) = present("required") {
        let param = format!("{path}.required");
        let Some(names) = v.as_array() else {
            return Err(invalid_schema(
                format!(
                    "'required' must be an array of property names, got {}",
                    json_kind(v)
                ),
                &param,
            ));
        };
        for (i, name) in names.iter().enumerate() {
            if !name.is_string() {
                return Err(invalid_schema(
                    format!(
                        "'required' entries must be strings, got {}",
                        json_kind(name)
                    ),
                    &format!("{param}[{i}]"),
                ));
            }
        }
    }

    if let Some(v) = present("enum") {
        if !v.is_array() {
            return Err(invalid_schema(
                format!("'enum' must be an array, got {}", json_kind(v)),
                &format!("{path}.enum"),
            ));
        }
    }

    for &kw in SCHEMA_ARRAY_KEYWORDS {
        if let Some(v) = present(kw) {
            let param = format!("{path}.{kw}");
            match v.as_array() {
                Some(items) if items.is_empty() => {
                    return Err(invalid_schema(
                        format!("'{kw}' must be a non-empty array of schemas"),
                        &param,
                    ));
                }
                Some(_) => {}
                None => {
                    return Err(invalid_schema(
                        format!(
                            "'{kw}' must be a non-empty array of schemas, got {}",
                            json_kind(v)
                        ),
                        &param,
                    ));
                }
            }
        }
    }

    if let Some(v) = present("prefixItems") {
        if !v.is_array() {
            return Err(invalid_schema(
                format!(
                    "'prefixItems' must be an array of schemas, got {}",
                    json_kind(v)
                ),
                &format!("{path}.prefixItems"),
            ));
        }
    }

    for &kw in NON_NEGATIVE_INTEGER_KEYWORDS {
        if let Some(v) = present(kw) {
            // ``5.0`` counts as an integer, as JSON Schema validators read it.
            let ok = v
                .as_f64()
                .is_some_and(|n| n >= 0.0 && n.fract() == 0.0 && n.is_finite());
            if !ok {
                return Err(invalid_schema(
                    format!("'{kw}' must be a non-negative integer, got {v}"),
                    &format!("{path}.{kw}"),
                ));
            }
        }
    }

    for &kw in NUMBER_KEYWORDS {
        if let Some(v) = present(kw) {
            if !v.is_number() {
                return Err(invalid_schema(
                    format!("'{kw}' must be a number, got {}", json_kind(v)),
                    &format!("{path}.{kw}"),
                ));
            }
        }
    }
    if let Some(v) = present("multipleOf").and_then(Value::as_f64) {
        if v <= 0.0 {
            return Err(invalid_schema(
                format!("'multipleOf' must be greater than 0, got {v}"),
                &format!("{path}.multipleOf"),
            ));
        }
    }

    if let Some(v) = present("pattern") {
        if !v.is_string() {
            return Err(invalid_schema(
                format!("'pattern' must be a string, got {}", json_kind(v)),
                &format!("{path}.pattern"),
            ));
        }
    }

    Ok(())
}

fn is_schema_nesting_key(key: &str) -> bool {
    matches!(
        key,
        "properties"
            | "patternProperties"
            | "additionalProperties"
            | "unevaluatedProperties"
            | "items"
            | "prefixItems"
            | "contains"
            | "propertyNames"
            | "oneOf"
            | "anyOf"
            | "allOf"
            | "not"
            | "definitions"
            | "$defs"
            | "dependentSchemas"
            | "if"
            | "then"
            | "else"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handlers::test_support::schema_with_shared_definition;
    use axum::body::to_bytes;
    use serde_json::json;

    async fn err_body(resp: Response) -> serde_json::Value {
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    fn ok_or_panic(r: GrammarParseResult) -> GrammarSpec {
        match r {
            GrammarParseResult::Ok(g) => g,
            GrammarParseResult::Err(_) => panic!("expected Ok"),
        }
    }

    async fn err_or_panic(r: GrammarParseResult) -> serde_json::Value {
        match r {
            GrammarParseResult::Ok(_) => panic!("expected Err"),
            GrammarParseResult::Err(resp) => err_body(resp).await,
        }
    }

    #[test]
    fn test_parse_grammar_accepts_small_json_schema() {
        let v = json!({
            "json_schema": {"type": "object", "properties": {"x": {"type": "integer"}}},
            "label": "tiny",
        });
        let spec = ok_or_panic(parse_grammar(&v));
        match spec {
            GrammarSpec::JsonSchema { label, .. } => {
                assert_eq!(label.as_deref(), Some("tiny"));
            }
            other => panic!("expected JsonSchema, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_grammar_accepts_unsupported_keyword_names_as_properties() {
        let v = json!({
            "json_schema": {
                "type": "object",
                "properties": {
                    "$dynamicRef": {"type": "string"},
                    "if": {"type": "string"},
                    "then": {"type": "integer"},
                    "else": {"type": "boolean"},
                    "unevaluatedProperties": {"type": "number"},
                    "dependentSchemas": {"type": "array", "items": {"type": "string"}},
                },
                "patternProperties": {
                    "^if$": {"type": "string"},
                },
            }
        });

        let spec = ok_or_panic(parse_grammar(&v));
        let GrammarSpec::JsonSchema { value, .. } = spec else {
            panic!("expected JsonSchema");
        };
        assert_eq!(value["properties"]["if"]["type"], "string");
        assert_eq!(
            value["properties"]["dependentSchemas"]["items"]["type"],
            "string"
        );
        assert_eq!(value["patternProperties"]["^if$"]["type"], "string");
    }

    #[test]
    fn test_parse_grammar_accepts_unsupported_keyword_names_as_definition_keys() {
        let v = json!({
            "json_schema": {
                "$defs": {
                    "if": {"type": "string"},
                    "then": {"type": "integer"},
                },
                "definitions": {
                    "else": {"type": "boolean"},
                },
                "type": "object",
                "properties": {
                    "condition": {"$ref": "#/$defs/if"},
                    "fallback": {"$ref": "#/definitions/else"},
                },
            }
        });

        let spec = ok_or_panic(parse_grammar(&v));
        let GrammarSpec::JsonSchema { value, .. } = spec else {
            panic!("expected JsonSchema");
        };
        assert_eq!(value["properties"]["condition"]["type"], "string");
        assert_eq!(value["properties"]["fallback"]["type"], "boolean");
    }

    #[test]
    fn test_parse_grammar_accepts_keyword_names_in_annotation_payloads() {
        let v = json!({
            "json_schema": {
                "type": "string",
                "default": {"if": "annotation data"},
                "examples": [{"then": "annotation data"}],
                "x-metadata": {"else": {"dependentSchemas": true}},
            }
        });

        let _ = ok_or_panic(parse_grammar(&v));
    }

    #[test]
    fn test_parse_grammar_accepts_small_regex() {
        let v = json!({"regex": r"[A-Z]{3}-\d{4}"});
        let spec = ok_or_panic(parse_grammar(&v));
        match spec {
            GrammarSpec::Regex { value, .. } => assert_eq!(value, r"[A-Z]{3}-\d{4}"),
            other => panic!("expected Regex, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_parse_grammar_rejects_oversized_payload() {
        // Build a schema strictly larger than the cap.
        let mut props = serde_json::Map::new();
        // Each property contributes ~30 bytes; (64 * 1024 / 30) ≈ 2240
        // is comfortably above. Pad to be safe.
        for i in 0..4000 {
            props.insert(
                format!("field_{i:05}"),
                json!({"type": "string", "description": "filler"}),
            );
        }
        let v = json!({"json_schema": {"type": "object", "properties": props}});
        let body = err_or_panic(parse_grammar(&v)).await;
        assert_eq!(body["error"]["code"], "invalid_request");
        assert_eq!(body["error"]["param"], "grammar");
        let msg = body["error"]["message"].as_str().unwrap_or("");
        assert!(msg.contains("exceeds limit"), "msg: {msg}");
    }

    #[tokio::test]
    async fn test_parse_grammar_rejects_deeply_nested_schema() {
        // Build a schema with depth > MAX_SCHEMA_DEPTH.
        let mut leaf = json!({"type": "string"});
        for _ in 0..(MAX_SCHEMA_DEPTH + 5) {
            leaf = json!({"type": "object", "properties": {"x": leaf}});
        }
        let v = json!({"json_schema": leaf});
        let body = err_or_panic(parse_grammar(&v)).await;
        assert_eq!(body["error"]["code"], "invalid_request");
        let param = body["error"]["param"].as_str().unwrap_or("");
        assert!(
            param.starts_with("grammar.json_schema"),
            "expected depth error path under grammar.json_schema, got {param}"
        );
    }

    fn deeply_nested_unknown_value(depth: usize) -> Value {
        let mut value = json!("leaf");
        for _ in 0..depth {
            value = json!({"x-annotation": value});
        }
        value
    }

    #[tokio::test]
    async fn test_dereference_schema_refs_rejects_raw_traversal_depth() {
        let schema = deeply_nested_unknown_value(MAX_SCHEMA_TRAVERSAL_DEPTH + 1);
        let rejection = dereference_schema_refs(&schema, "grammar.json_schema")
            .expect_err("raw traversal depth should reject during dereference");
        let body = err_body(rejection.into_response()).await;

        assert_eq!(body["error"]["code"], "invalid_request");
        assert!(body["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("traversal depth exceeds limit")));
    }

    #[tokio::test]
    async fn test_walk_schema_rejects_raw_traversal_depth() {
        let schema = deeply_nested_unknown_value(MAX_SCHEMA_TRAVERSAL_DEPTH + 1);
        let rejection = walk_schema(&schema, "grammar.json_schema", None)
            .expect_err("raw traversal depth should reject during validation");
        let body = err_body(rejection.into_response()).await;

        assert_eq!(body["error"]["code"], "invalid_request");
        assert!(body["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("traversal depth exceeds limit")));
    }

    #[test]
    fn test_parse_grammar_dereferences_internal_dollar_ref() {
        let v = json!({
            "json_schema": {
                "type": "object",
                "$defs": {
                    "Foo": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    }
                },
                "properties": {"a": {"$ref": "#/$defs/Foo"}},
            }
        });
        let spec = ok_or_panic(parse_grammar(&v));
        let GrammarSpec::JsonSchema { value, .. } = spec else {
            panic!("expected JsonSchema");
        };
        assert_eq!(
            value["properties"]["a"]["properties"]["name"]["type"],
            "string"
        );
        assert!(
            value.get("$defs").is_none(),
            "$defs should be stripped after inlining"
        );
        let encoded = serde_json::to_string(&value).unwrap();
        assert!(
            !encoded.contains("\"$ref\""),
            "schema should be fully dereferenced: {encoded}"
        );
    }

    #[test]
    fn test_parse_grammar_preserves_dollar_ref_sibling_constraints() {
        let v = json!({
            "json_schema": {
                "type": "object",
                "$defs": {
                    "Foo": {
                        "type": "object",
                        "properties": {"base": {"type": "string"}},
                        "required": ["base"],
                    }
                },
                "properties": {
                    "item": {
                        "$ref": "#/$defs/Foo",
                        "type": "object",
                        "properties": {"extra": {"type": "integer"}},
                        "required": ["extra"],
                    }
                },
            }
        });
        let spec = ok_or_panic(parse_grammar(&v));
        let GrammarSpec::JsonSchema { value, .. } = spec else {
            panic!("expected JsonSchema");
        };
        let all_of = value["properties"]["item"]["allOf"]
            .as_array()
            .expect("ref sibling schema should be represented as allOf");
        assert_eq!(all_of.len(), 2);
        assert_eq!(all_of[0]["properties"]["base"]["type"], "string");
        assert_eq!(all_of[0]["required"], json!(["base"]));
        assert_eq!(all_of[1]["properties"]["extra"]["type"], "integer");
        assert_eq!(all_of[1]["required"], json!(["extra"]));
        let encoded = serde_json::to_string(&value).unwrap();
        assert!(
            !encoded.contains("\"$ref\"") && !encoded.contains("\"$defs\""),
            "schema should be fully dereferenced: {encoded}"
        );
    }

    #[test]
    fn test_parse_grammar_dereferences_pydantic_openai_style_schema() {
        let v = json!({
            "json_schema": {
                "$defs": {
                    "Step": {
                        "type": "object",
                        "properties": {
                            "explanation": {"type": "string"},
                            "output": {"type": "string"},
                        },
                        "required": ["explanation", "output"],
                        "additionalProperties": false,
                    }
                },
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/Step"},
                    },
                    "single": {
                        "$ref": "#/$defs/Step",
                        "description": "one step",
                    },
                    "final_answer": {"type": "string"},
                },
                "required": ["steps", "single", "final_answer"],
                "additionalProperties": false,
            }
        });
        let spec = ok_or_panic(parse_grammar(&v));
        let GrammarSpec::JsonSchema { value, .. } = spec else {
            panic!("expected JsonSchema");
        };
        assert_eq!(
            value["properties"]["steps"]["items"]["properties"]["explanation"]["type"],
            "string"
        );
        assert_eq!(
            value["properties"]["steps"]["items"]["additionalProperties"],
            false
        );
        let single_all_of = value["properties"]["single"]["allOf"]
            .as_array()
            .expect("Pydantic ref sibling should be preserved as allOf");
        assert_eq!(
            single_all_of[0]["properties"]["explanation"]["type"],
            "string"
        );
        assert_eq!(single_all_of[1]["description"], "one step");
        assert_eq!(value["additionalProperties"], false);
        let encoded = serde_json::to_string(&value).unwrap();
        assert!(
            !encoded.contains("\"$ref\"") && !encoded.contains("\"$defs\""),
            "schema should be fully dereferenced: {encoded}"
        );
    }

    #[tokio::test]
    async fn test_parse_grammar_rejects_external_dollar_ref() {
        let v = json!({
            "json_schema": {
                "type": "object",
                "properties": {"a": {"$ref": "https://example.com/schemas/foo.json"}},
            }
        });
        let body = err_or_panic(parse_grammar(&v)).await;
        assert_eq!(body["error"]["code"], "unsupported_field");
        assert_eq!(
            body["error"]["param"],
            "grammar.json_schema.properties.a.$ref"
        );
        let msg = body["error"]["message"].as_str().unwrap_or("");
        assert!(
            msg.contains("external"),
            "msg should mention external refs: {msg}"
        );
    }

    #[tokio::test]
    async fn test_parse_grammar_rejects_recursive_dollar_ref_cycle() {
        let v = json!({
            "json_schema": {
                "$defs": {
                    "A": {"$ref": "#/$defs/B"},
                    "B": {"$ref": "#/$defs/A"},
                },
                "properties": {"a": {"$ref": "#/$defs/A"}},
            }
        });
        let body = err_or_panic(parse_grammar(&v)).await;
        assert_eq!(body["error"]["code"], "invalid_request");
        assert_eq!(
            body["error"]["param"],
            "grammar.json_schema.properties.a.$ref"
        );
        let msg = body["error"]["message"].as_str().unwrap_or("");
        assert!(msg.contains("cycle"), "msg should mention cycle: {msg}");
    }

    #[tokio::test]
    async fn test_parse_grammar_rejects_each_unsupported_keyword() {
        for kw in [
            "$dynamicRef",
            "if",
            "then",
            "else",
            "unevaluatedProperties",
            "dependentSchemas",
        ] {
            let mut schema = serde_json::Map::new();
            schema.insert("type".to_string(), json!("object"));
            schema.insert(kw.to_string(), json!({"x": true}));
            let v = json!({"json_schema": Value::Object(schema)});
            let body = err_or_panic(parse_grammar(&v)).await;
            assert_eq!(
                body["error"]["code"], "unsupported_field",
                "keyword {kw} should reject"
            );
            assert_eq!(
                body["error"]["param"],
                format!("grammar.json_schema.{kw}"),
                "param path for {kw}"
            );
        }
    }

    #[tokio::test]
    async fn test_parse_grammar_rejects_unsupported_keyword_in_nested_schema_context() {
        let v = json!({
            "json_schema": {
                "type": "object",
                "properties": {
                    "result": {
                        "allOf": [
                            {
                                "type": "object",
                                "if": {"properties": {"enabled": {"const": true}}},
                            }
                        ]
                    }
                }
            }
        });

        let body = err_or_panic(parse_grammar(&v)).await;
        assert_eq!(body["error"]["code"], "unsupported_field");
        assert_eq!(
            body["error"]["param"],
            "grammar.json_schema.properties.result.allOf[0].if"
        );
    }

    #[tokio::test]
    async fn test_parse_grammar_rejects_oversized_regex() {
        let big = "a".repeat(MAX_REGEX_LEN + 1);
        let v = json!({"regex": big});
        let body = err_or_panic(parse_grammar(&v)).await;
        assert_eq!(body["error"]["code"], "invalid_request");
        assert_eq!(body["error"]["param"], "grammar.regex");
    }

    #[tokio::test]
    async fn test_parse_grammar_rejects_mutual_exclusivity_violation() {
        let v = json!({
            "json_schema": {"type": "object"},
            "regex": "abc",
        });
        let body = err_or_panic(parse_grammar(&v)).await;
        assert_eq!(body["error"]["code"], "invalid_request");
        assert_eq!(body["error"]["param"], "grammar");
        let msg = body["error"]["message"].as_str().unwrap_or("");
        assert!(
            msg.contains("mutually exclusive"),
            "msg should mention mutual exclusivity: {msg}"
        );
    }

    #[test]
    fn test_parse_grammar_accepts_small_ebnf() {
        let v = json!({
            "ebnf": "root ::= \"hello\" | \"goodbye\"",
            "label": "greeting",
        });
        let spec = ok_or_panic(parse_grammar(&v));
        match spec {
            GrammarSpec::Ebnf { value, label, .. } => {
                assert!(value.contains("root ::="));
                assert_eq!(label.as_deref(), Some("greeting"));
            }
            other => panic!("expected Ebnf, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_parse_grammar_rejects_oversized_ebnf() {
        let big = "a".repeat(MAX_EBNF_LEN + 1);
        let v = json!({"ebnf": big});
        let body = err_or_panic(parse_grammar(&v)).await;
        assert_eq!(body["error"]["code"], "invalid_request");
        assert_eq!(body["error"]["param"], "grammar.ebnf");
    }

    #[tokio::test]
    async fn test_parse_grammar_rejects_non_string_ebnf() {
        let v = json!({"ebnf": 42});
        let body = err_or_panic(parse_grammar(&v)).await;
        assert_eq!(body["error"]["param"], "grammar.ebnf");
    }

    #[tokio::test]
    async fn test_parse_grammar_rejects_ebnf_plus_other_variant() {
        let v = json!({
            "ebnf": "root ::= \"a\"",
            "regex": "[a-z]+",
        });
        let body = err_or_panic(parse_grammar(&v)).await;
        assert_eq!(body["error"]["code"], "invalid_request");
        assert_eq!(body["error"]["param"], "grammar");
    }

    #[tokio::test]
    async fn test_parse_grammar_rejects_empty_object() {
        let v = json!({});
        let body = err_or_panic(parse_grammar(&v)).await;
        assert_eq!(body["error"]["param"], "grammar");
    }

    #[tokio::test]
    async fn test_parse_grammar_rejects_non_object() {
        let v = json!("oops");
        let body = err_or_panic(parse_grammar(&v)).await;
        assert_eq!(body["error"]["param"], "grammar");
    }

    #[tokio::test]
    async fn test_parse_grammar_rejects_non_string_regex() {
        let v = json!({"regex": 42});
        let body = err_or_panic(parse_grammar(&v)).await;
        assert_eq!(body["error"]["param"], "grammar.regex");
    }

    #[test]
    fn test_parse_grammar_passes_through_label_and_strict() {
        let v = json!({
            "json_schema": {"type": "string"},
            "label": "name_v1",
            "strict": true,
        });
        match ok_or_panic(parse_grammar(&v)) {
            GrammarSpec::JsonSchema { label, strict, .. } => {
                assert_eq!(label.as_deref(), Some("name_v1"));
                assert_eq!(strict, Some(true));
            }
            other => panic!("expected JsonSchema, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_parse_grammar_rejects_strict_ebnf() {
        let v = json!({"ebnf": "root ::= \"yes\" | \"no\"", "strict": true});
        let body = err_or_panic(parse_grammar(&v)).await;
        assert_eq!(body["error"]["param"], "grammar.strict");
        assert_eq!(body["error"]["code"], "unsupported_field");

        let v = json!({"ebnf": "root ::= \"yes\" | \"no\"", "strict": false});
        match ok_or_panic(parse_grammar(&v)) {
            GrammarSpec::Ebnf { strict, .. } => assert_eq!(strict, Some(false)),
            other => panic!("expected Ebnf, got {other:?}"),
        }
    }

    // ── capability gate ────────────────────────────────────────────

    #[tokio::test]
    async fn test_check_capability_accepts_listed_kind() {
        let g = GrammarSpec::JsonSchema {
            value: json!({"type": "string"}),
            label: None,
            strict: None,
        };
        let caps = vec!["json_schema".to_string(), "regex".to_string()];
        assert!(check_capability(&g, Some(&caps), "m").is_ok());
    }

    #[tokio::test]
    async fn test_check_capability_rejects_unlisted_kind() {
        let g = GrammarSpec::Regex {
            value: "[a-z]+".to_string(),
            label: None,
            strict: None,
        };
        let caps = vec!["json_schema".to_string()];
        let resp = check_capability(&g, Some(&caps), "Qwen/X").expect_err("expected reject");
        let body = err_body(resp).await;
        assert_eq!(body["error"]["code"], "unsupported_field");
        assert_eq!(body["error"]["param"], "grammar.regex");
        let msg = body["error"]["message"].as_str().unwrap_or("");
        assert!(msg.contains("Qwen/X"), "msg should mention model: {msg}");
        assert!(msg.contains("regex"), "msg should mention kind: {msg}");
    }

    #[tokio::test]
    async fn test_check_capability_accepts_ebnf_when_listed() {
        let g = GrammarSpec::Ebnf {
            value: "root ::= \"a\"".to_string(),
            label: None,
            strict: None,
        };
        let caps = vec!["ebnf".to_string()];
        assert!(check_capability(&g, Some(&caps), "m").is_ok());
    }

    #[tokio::test]
    async fn test_check_capability_rejects_ebnf_when_unlisted() {
        let g = GrammarSpec::Ebnf {
            value: "root ::= \"a\"".to_string(),
            label: None,
            strict: None,
        };
        let caps = vec!["json_schema".to_string(), "regex".to_string()];
        let resp = check_capability(&g, Some(&caps), "Qwen/X").expect_err("expected reject");
        let body = err_body(resp).await;
        assert_eq!(body["error"]["code"], "unsupported_field");
        assert_eq!(body["error"]["param"], "grammar.ebnf");
    }

    #[tokio::test]
    async fn test_check_capability_rejects_empty_list() {
        let g = GrammarSpec::Regex {
            value: "[a-z]+".to_string(),
            label: None,
            strict: None,
        };
        let caps: Vec<String> = Vec::new();
        assert!(check_capability(&g, Some(&caps), "m").is_err());
    }

    #[tokio::test]
    async fn test_check_capability_rejects_none_capabilities() {
        // Model has no ``generate`` task at all — anything grammar-shaped
        // must reject.
        let g = GrammarSpec::JsonSchema {
            value: json!({}),
            label: None,
            strict: None,
        };
        let resp = check_capability(&g, None, "m").expect_err("expected reject");
        let body = err_body(resp).await;
        let msg = body["error"]["message"].as_str().unwrap_or("");
        assert!(msg.contains("does not support grammar"), "msg: {msg}");
    }

    /// Depth budget is generous enough for realistic extraction schemas
    /// (4-5 levels of objects/arrays). Exercise a moderate schema to
    /// guard against an off-by-one that would falsely reject sensible
    /// inputs.
    #[test]
    fn test_parse_grammar_accepts_realistic_extraction_schema() {
        let v = json!({
            "json_schema": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "qty": {"type": "integer"},
                            },
                            "required": ["name", "qty"],
                        }
                    },
                    "total": {"type": "number"}
                },
                "required": ["items"],
                "additionalProperties": false
            }
        });
        let _ = ok_or_panic(parse_grammar(&v));
    }

    const TOOL_PATH: &str = "tools[0].function.parameters";

    fn tool_rejection(schema: &Value) -> SchemaRejection {
        validate_tool_parameters(schema, TOOL_PATH)
            .expect_err("expected the tool schema to be refused")
    }

    #[tokio::test]
    async fn test_tool_parameters_accept_refs_that_expand_past_the_node_cap() {
        let schema = schema_with_shared_definition(150, 40, 1);
        assert!(serde_json::to_vec(&schema).unwrap().len() < MAX_GRAMMAR_BYTES);

        validate_tool_parameters(&schema, TOOL_PATH).expect("tool schema should be accepted");

        let body = err_or_panic(parse_grammar(&json!({"json_schema": schema}))).await;
        assert_eq!(
            body["error"]["message"],
            format!("JSON Schema node count exceeds limit ({MAX_SCHEMA_NODES})")
        );
    }

    #[tokio::test]
    async fn test_tool_parameters_accept_refs_that_expand_past_the_byte_cap() {
        let schema = schema_with_shared_definition(40, 2, 2000);
        assert!(serde_json::to_vec(&schema).unwrap().len() < MAX_GRAMMAR_BYTES);

        validate_tool_parameters(&schema, TOOL_PATH).expect("tool schema should be accepted");

        let body = err_or_panic(parse_grammar(&json!({"json_schema": schema}))).await;
        let message = body["error"]["message"].as_str().unwrap_or("");
        assert!(
            message.starts_with("grammar payload ")
                && message.ends_with(&format!("exceeds limit ({MAX_GRAMMAR_BYTES} bytes)")),
            "message: {message}"
        );
    }

    #[tokio::test]
    async fn test_tool_parameters_accept_a_recursive_definition() {
        let schema = json!({
            "type": "object",
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {"children": {"type": "array", "items": {"$ref": "#/$defs/Node"}}},
                }
            },
            "properties": {"root": {"$ref": "#/$defs/Node"}},
        });

        validate_tool_parameters(&schema, TOOL_PATH).expect("tool schema should be accepted");

        let body = err_or_panic(parse_grammar(&json!({"json_schema": schema}))).await;
        assert!(body["error"]["message"]
            .as_str()
            .is_some_and(|message| message.starts_with("recursive '$ref' cycle detected")));
    }

    #[test]
    fn test_tool_parameters_reject_an_unresolved_ref() {
        let rejection = tool_rejection(&json!({
            "type": "object",
            "properties": {"a": {"$ref": "#/$defs/Missing"}},
        }));
        assert_eq!(
            rejection.message,
            "unresolved internal '$ref' \"#/$defs/Missing\""
        );
        assert_eq!(rejection.param, format!("{TOOL_PATH}.properties.a.$ref"));
        assert_eq!(rejection.code, oai_code::INVALID_REQUEST);
    }

    #[test]
    fn test_tool_parameters_reject_an_external_ref() {
        let rejection = tool_rejection(&json!({
            "type": "object",
            "properties": {"a": {"$ref": "https://example.com/schema.json"}},
        }));
        assert_eq!(rejection.message, "external '$ref' is not supported");
        assert_eq!(rejection.param, format!("{TOOL_PATH}.properties.a.$ref"));
        assert_eq!(rejection.code, oai_code::UNSUPPORTED_FIELD);
    }

    #[test]
    fn test_tool_parameters_reject_an_unsupported_keyword_inside_a_definition() {
        let rejection = tool_rejection(&json!({
            "type": "object",
            "$defs": {"Shared": {"type": "object", "if": {"required": ["a"]}}},
            "properties": {"a": {"$ref": "#/$defs/Shared"}},
        }));
        assert_eq!(
            rejection.message,
            "JSON Schema keyword 'if' is not supported"
        );
        assert_eq!(rejection.param, format!("{TOOL_PATH}.$defs.Shared.if"));
        assert_eq!(rejection.code, oai_code::UNSUPPORTED_FIELD);
    }

    #[test]
    fn test_tool_parameters_reject_an_oversized_schema() {
        let schema = json!({"type": "object", "description": "d".repeat(MAX_GRAMMAR_BYTES)});
        let size = serde_json::to_vec(&schema).unwrap().len();

        let rejection = tool_rejection(&schema);
        assert_eq!(
            rejection.message,
            format!("JSON Schema {size} bytes exceeds limit ({MAX_GRAMMAR_BYTES} bytes)")
        );
        assert_eq!(rejection.param, TOOL_PATH);
    }

    #[test]
    fn test_tool_parameters_reject_a_deep_schema() {
        let mut schema = json!({"type": "string"});
        for _ in 0..=MAX_SCHEMA_DEPTH {
            schema = json!({"type": "object", "properties": {"x": schema}});
        }

        let rejection = tool_rejection(&schema);
        assert_eq!(
            rejection.message,
            format!("JSON Schema depth exceeds limit ({MAX_SCHEMA_DEPTH})")
        );
        assert!(
            rejection.param.starts_with(TOOL_PATH),
            "param: {}",
            rejection.param
        );
    }

    #[test]
    fn test_tool_parameters_reject_too_many_nodes() {
        let schema = json!({"type": "integer", "enum": vec![0; MAX_SCHEMA_NODES]});
        assert!(serde_json::to_vec(&schema).unwrap().len() < MAX_GRAMMAR_BYTES);

        let rejection = tool_rejection(&schema);
        assert_eq!(
            rejection.message,
            format!("JSON Schema node count exceeds limit ({MAX_SCHEMA_NODES})")
        );
    }

    /// Schemas that every JSON Schema draft refuses, with the ``param``
    /// each refusal names (relative to the schema root ``p``).
    fn structurally_invalid_schemas() -> Vec<(Value, &'static str, &'static str)> {
        vec![
            (
                json!({"type": "objekt"}),
                "p.type",
                "'type' \"objekt\" is not a JSON Schema type",
            ),
            (
                json!({"type": 3}),
                "p.type",
                "'type' must be a string or an array of strings",
            ),
            (
                json!({"type": []}),
                "p.type",
                "'type' must not be an empty array",
            ),
            (
                json!({"type": ["string", "nul"]}),
                "p.type[1]",
                "'type' \"nul\" is not a JSON Schema type",
            ),
            (
                json!({"type": ["string", 1]}),
                "p.type[1]",
                "'type' entries must be strings",
            ),
            (
                json!({"type": "object", "properties": [{"a": {"type": "string"}}]}),
                "p.properties",
                "'properties' must be an object mapping names to schemas",
            ),
            (
                json!({"type": "object", "properties": {"a": "string"}}),
                "p.properties.a",
                "a JSON Schema must be an object or a boolean",
            ),
            (
                json!({"type": "object", "properties": {"a": {"type": "string"}}, "required": "a"}),
                "p.required",
                "'required' must be an array of property names",
            ),
            (
                json!({"type": "object", "properties": {"a": {"type": "string", "required": true}}}),
                "p.properties.a.required",
                "'required' must be an array of property names",
            ),
            (
                json!({"type": "object", "required": ["a", 1]}),
                "p.required[1]",
                "'required' entries must be strings",
            ),
            (
                json!({"type": "string", "enum": "a"}),
                "p.enum",
                "'enum' must be an array",
            ),
            (
                json!({"anyOf": []}),
                "p.anyOf",
                "'anyOf' must be a non-empty array of schemas",
            ),
            (
                json!({"oneOf": {"type": "string"}}),
                "p.oneOf",
                "'oneOf' must be a non-empty array of schemas",
            ),
            (
                json!({"type": "string", "minLength": -1}),
                "p.minLength",
                "'minLength' must be a non-negative integer",
            ),
            (
                json!({"type": "array", "maxItems": 1.5}),
                "p.maxItems",
                "'maxItems' must be a non-negative integer",
            ),
            (
                json!({"type": "number", "minimum": "0"}),
                "p.minimum",
                "'minimum' must be a number",
            ),
            (
                json!({"type": "number", "multipleOf": 0}),
                "p.multipleOf",
                "'multipleOf' must be greater than 0",
            ),
            (
                json!({"type": "string", "pattern": 1}),
                "p.pattern",
                "'pattern' must be a string",
            ),
            (
                json!({"type": "array", "items": "string"}),
                "p.items",
                "a JSON Schema must be an object or a boolean",
            ),
            (
                json!("object"),
                "p",
                "a JSON Schema must be an object or a boolean",
            ),
            (
                json!({"type": "object", "$defs": 7}),
                "p.$defs",
                "'$defs' must be an object mapping names to schemas",
            ),
            (
                json!({"type": "object", "definitions": ["a"]}),
                "p.definitions",
                "'definitions' must be an object mapping names to schemas",
            ),
            (
                json!([]),
                "p",
                "a JSON Schema must be an object or a boolean",
            ),
            (
                json!({"type": "object", "properties": {"a": []}}),
                "p.properties.a",
                "a JSON Schema must be an object or a boolean",
            ),
            (
                json!({"allOf": [[]]}),
                "p.allOf[0]",
                "a JSON Schema must be an object or a boolean",
            ),
            (
                json!({"not": [{"type": "string"}]}),
                "p.not",
                "a JSON Schema must be an object or a boolean",
            ),
            (
                json!({"type": "array", "items": [[{"type": "string"}]]}),
                "p.items[0]",
                "a JSON Schema must be an object or a boolean",
            ),
            (
                json!({"type": "object", "properties": {"a": {"$ref": "#/$defs/A"}}, "$defs": {"A": {"type": "strin"}}}),
                "p.$defs.A.type",
                "'type' \"strin\" is not a JSON Schema type",
            ),
            (
                json!({"type": "object", "properties": {"a": {"type": "array", "items": {"anyOf": [{"type": "string"}, {"type": "object", "properties": {"b": {"type": "intger"}}}]}}}}),
                "p.properties.a.items.anyOf[1].properties.b.type",
                "'type' \"intger\" is not a JSON Schema type",
            ),
        ]
    }

    /// Valid schemas from common clients that must keep passing.
    fn structurally_valid_schemas() -> Vec<Value> {
        vec![
            json!({}),
            json!(true),
            json!({"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"], "additionalProperties": false}),
            json!({"type": "object", "properties": {"u": {"type": ["string", "null"], "enum": ["c", "f", null]}}}),
            json!({"type": "object", "properties": {"a": {"$ref": "#/$defs/A"}}, "$defs": {"A": {"type": "object", "properties": {"b": {"type": "integer", "minimum": 0}}}}}),
            json!({"type": "object", "properties": {"a": {"$ref": "#/definitions/A"}}, "definitions": {"A": {"type": "string", "minLength": 1, "maxLength": 5.0, "pattern": "^x"}}}),
            json!({"anyOf": [{"type": "string"}, {"type": "null"}], "default": null, "title": "T"}),
            json!({"type": "array", "items": [{"type": "string"}, {"type": "number"}], "minItems": 2, "additionalItems": false}),
            json!({"type": "number", "exclusiveMinimum": true, "minimum": 0, "multipleOf": 0.5}),
            // Unknown keywords and annotations pass, whatever they hold;
            // property names that look like keywords are names, not keywords.
            json!({"type": "object", "x-extra": {"type": "objekt"}, "nullable": true, "properties": {"type": {"type": "string"}, "required": {"type": "boolean"}, "enum": {"type": "array"}}}),
            // ``null`` keyword values read as absent.
            json!({"type": "object", "properties": {"a": {"type": "string"}}, "required": null, "items": null}),
            json!({"type": "string", "format": "custom", "description": "d", "examples": ["x"], "const": "x"}),
            // A definition may hold an array that a ``$ref`` indexes into.
            json!({"type": "object", "$defs": {"variants": [{"type": "string"}, {"type": "integer"}]}, "properties": {"a": {"$ref": "#/$defs/variants/1"}}}),
        ]
    }

    #[test]
    fn test_tool_parameters_reject_structurally_invalid_schemas() {
        for (schema, param, message) in structurally_invalid_schemas() {
            let rejection = validate_tool_parameters(&schema, "p")
                .expect_err(&format!("{schema} should be refused"));
            assert_eq!(rejection.param, param, "{schema}");
            assert_eq!(rejection.code, oai_code::INVALID_REQUEST, "{schema}");
            assert!(
                rejection.message.starts_with(message),
                "{schema}: {}",
                rejection.message
            );
        }
    }

    #[test]
    fn test_tool_parameters_accept_structurally_valid_schemas() {
        for schema in structurally_valid_schemas() {
            if let Err(rejection) = validate_tool_parameters(&schema, "p") {
                panic!("{schema} refused: {rejection:?}");
            }
        }
    }

    #[tokio::test]
    async fn test_parse_grammar_rejects_structurally_invalid_schemas() {
        for (schema, param, message) in structurally_invalid_schemas() {
            // A refused ``$defs`` entry is reported where it is referenced,
            // since the grammar walks the schema with ``$ref`` inlined.
            let param = param
                .replacen('p', "grammar.json_schema", 1)
                .replace("$defs.A", "properties.a");
            let body = err_or_panic(parse_grammar(&json!({"json_schema": schema}))).await;
            assert_eq!(body["error"]["param"], param, "{schema}");
            assert_eq!(body["error"]["code"], "invalid_request", "{schema}");
            let got = body["error"]["message"].as_str().unwrap();
            assert!(got.starts_with(message), "{schema}: {got}");
        }
    }

    #[test]
    fn test_parse_grammar_accepts_structurally_valid_schemas() {
        for schema in structurally_valid_schemas() {
            if let GrammarParseResult::Err(_) = parse_grammar(&json!({"json_schema": schema})) {
                panic!("{schema} refused");
            }
        }
    }

    #[tokio::test]
    async fn test_parse_grammar_at_names_the_given_schema_path() {
        let v = json!({"json_schema": {"type": "objekt"}});
        let body = err_or_panic(parse_grammar_at(&v, "response_format.json_schema.schema")).await;
        assert_eq!(
            body["error"]["param"],
            "response_format.json_schema.schema.type"
        );
    }
}
