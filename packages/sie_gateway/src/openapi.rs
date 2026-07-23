use axum::http::header;
use axum::response::IntoResponse;
use serde::{Deserialize, Serialize};
use serde_json::json;
use serde_json::Value;
use std::path::Path;
use std::sync::LazyLock;
use utoipa::{OpenApi, ToSchema};

static OPENAPI_DOC: LazyLock<utoipa::openapi::OpenApi> = LazyLock::new(|| {
    let mut doc = ApiDoc::openapi();
    doc.info.version = env!("CARGO_PKG_VERSION").to_string();
    doc
});

static OPENAPI_JSON: LazyLock<String> = LazyLock::new(|| {
    let mut value = serde_json::to_value(&*OPENAPI_DOC).expect("OpenAPI document should serialize");
    apply_gateway_openapi_overrides(&mut value);
    serde_json::to_string_pretty(&value).expect("OpenAPI document should serialize") + "\n"
});

#[derive(OpenApi)]
#[openapi(
    info(
        title = "SIE Gateway",
        description = "Rust gateway API for SIE inference, pool coordination, and read-only runtime config.",
        version = "0.0.0"
    ),
    paths(
        crate::handlers::health::status_page,
        openapi_json,
        crate::handlers::health::healthz,
        crate::handlers::health::readyz,
        crate::handlers::health::health,
        crate::handlers::health::ws_cluster_status,
        crate::handlers::models::get_models,
        crate::handlers::models::get_model,
        crate::handlers::pools::create_pool,
        crate::handlers::pools::list_pools,
        crate::handlers::pools::get_pool,
        crate::handlers::pools::delete_pool,
        crate::handlers::pools::renew_pool,
        crate::handlers::config_api::get_model_configs,
        crate::handlers::config_api::get_model_config_or_status,
        get_model_config_status_doc,
        crate::handlers::config_api::get_bundle_configs,
        crate::handlers::config_api::get_bundle_config,
        crate::handlers::config_api::resolve_config,
        crate::handlers::proxy::proxy_encode,
        crate::handlers::proxy::proxy_openai_embeddings,
        crate::handlers::audio::proxy_openai_transcription,
        crate::handlers::proxy::proxy_rerank,
        crate::handlers::proxy::proxy_rerank_v2,
        crate::handlers::proxy::proxy_score,
        crate::handlers::proxy::proxy_extract,
        crate::handlers::proxy::proxy_generate,
        crate::handlers::proxy::proxy_chat,
        crate::handlers::proxy::proxy_completions,
        crate::handlers::proxy::proxy_responses,
        crate::handlers::proxy::proxy_moderations,
        docs_ui
    ),
    components(schemas(
        BundleConfigDocument,
        BundleConfigSummary,
        BundleConfigsResponse,
        ClusterSummary,
        ConfigModelDocument,
        ConfigModelSummary,
        ConfigModelsResponse,
        crate::handlers::pools::CreatePoolRequest,
        crate::queue::publisher::PendingGenerationGroup,
        crate::queue::publisher::PendingGenerationSnapshot,
        ErrorDetailCore,
        StandardApiError,
        GpuNotConfiguredDetail,
        GpuNotConfiguredError,
        GatewayModelLoadFailedDetail,
        GatewayModelLoadFailedResponse,
        GatewayErrorResponse,
        InferenceInternalServerErrorResponse,
        InferenceServiceUnavailableResponse,
        AllItemsFailedResponse,
        BundleRoutingConflictDetail,
        BundleConflictResponse,
        DocumentInput,
        DenseVector,
        EncodeParams,
        EncodeRequest,
        EncodeResponse,
        EncodeResult,
        Entity,
        ExtractItemError,
        ExtractParams,
        ExtractRequest,
        ExtractResponse,
        ExtractResult,
        ImageInput,
        HealthResponse,
        ItemInput,
        InferenceErrorDetail,
        MessageResponse,
        ModelAckBundleStatus,
        ModelCapabilitiesWire,
        ModelConfigStatusResponse,
        ModelInfoWire,
        ModelsResponse,
        OpenAiModelObject,
        ModelNotFoundDetail,
        ModelNotFoundResponse,
        ProfileInfoWire,
        crate::types::worker::ModelInfo,
        MultiVector,
        OpenAIEmbeddingDataEntry,
        OpenAIEmbeddingEncodingFormat,
        OpenAIEmbeddingInput,
        OpenAIEmbeddingRequest,
        OpenAIEmbeddingUsage,
        OpenAIEmbeddingVector,
        OpenAIEmbeddingsListResponse,
        OpenAITranscriptionRequest,
        OpenAITranscriptionResponse,
        OpenAITranscriptionResponseFormat,
        OpenAITranscriptionUsage,
        NativeGenerateImage,
        GenerateRequest,
        GenerateResponse,
        GenerateChunk,
        GenerateChunkError,
        GenerateUsage,
        ChatCompletionRequest,
        ChatCompletionMessage,
        ChatCompletionResponse,
        ChatCompletionChoice,
        ChatCompletionChoiceMessage,
        ChatCompletionUsage,
        CompletionsRequest,
        ResponsesRequest,
        OpenAIErrorBody,
        OpenAIErrorEnvelope,
        PoolListResponse,
        crate::types::pool::PoolSpec,
        crate::types::pool::PoolStatus,
        crate::types::pool::PoolState,
        Relation,
        ResolveBundleConflictDetail,
        ResolveBundleConflictResponse,
        ResolveModelNotFoundDetail,
        ResolveModelNotFoundResponse,
        ResolveConfigResponse,
        RerankDocument,
        RerankError,
        RerankOptions,
        RerankRequest,
        RerankV2Request,
        RerankResponse,
        RerankResult,
        ScoreUsage,
        ScoreEntry,
        crate::handlers::config_api::ResolveRequest,
        ScoreRequest,
        ScoreResponse,
        SparseVector,
        TimingInfo,
        crate::types::pool::AssignedWorker,
        crate::types::worker::WorkerInfo
    )),
    tags(
        (name = "health", description = "Gateway health, readiness, and status surfaces"),
        (name = "inference", description = "Queue-backed inference entrypoints"),
        (name = "models", description = "Models visible to this gateway replica"),
        (name = "pools", description = "Runtime pool coordination"),
        (name = "config", description = "Read-only gateway view of model and bundle config"),
        (name = "observability", description = "Metrics and streaming status"),
        (name = "docs", description = "API description")
    )
)]
pub struct ApiDoc;

#[utoipa::path(
    get,
    path = "/openapi.json",
    tag = "docs",
    responses((status = 200, description = "OpenAPI document"))
)]
pub async fn openapi_json() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "application/json")],
        OPENAPI_JSON.as_str(),
    )
}

/// Vendored Redoc standalone bundle — pinned **v2.5.0**, sha256
/// `0ec05be285ac885a330289b02f470e1bdbd2b6b3223a9fa213f24bf805a851d1`,
/// from `https://cdn.redoc.ly/redoc/v2.5.0/bundles/redoc.standalone.js`.
/// Inlined into the binary so `/docs` is fully self-contained: no runtime
/// CDN/egress dependency, works in air-gapped clusters. Treat the pin as a
/// tracked dependency — bump by re-vendoring the file and re-running the
/// `docs_*` tests.
const REDOC_BUNDLE: &str = include_str!("../assets/redoc.standalone.js");

/// Self-contained Redoc page. Renders the **live** `/openapi.json` at
/// request time (so it can never drift from the served spec) and loads the
/// vendored bundle from a sibling route. Both URLs are relative so the page
/// works unchanged at `/docs` and behind a sub-path ingress.
const REDOC_HTML: &str = r#"<!DOCTYPE html>
<html>
  <head>
    <title>SIE Gateway — API reference</title>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <style>body { margin: 0; padding: 0; }</style>
  </head>
  <body>
    <redoc spec-url="openapi.json"></redoc>
    <script src="docs/redoc.standalone.js"></script>
  </body>
</html>
"#;

/// Rendered, human-browsable API reference (Redoc) over the live
/// `/openapi.json`. Auth-exempt (documentation, like `/openapi.json`) and
/// read-only — no in-browser request console, so no token-leak surface.
#[utoipa::path(
    get,
    path = "/docs",
    tag = "docs",
    responses((status = 200, description = "Rendered API reference (Redoc)"))
)]
pub async fn docs_ui() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        REDOC_HTML,
    )
}

/// The vendored Redoc JS bundle referenced by [`docs_ui`]. Served from the
/// gateway itself (not a CDN) so `/docs` has no outbound dependency.
pub async fn redoc_asset() -> impl IntoResponse {
    (
        [(
            header::CONTENT_TYPE,
            "application/javascript; charset=utf-8",
        )],
        REDOC_BUNDLE,
    )
}

#[cfg(test)]
fn openapi_document() -> utoipa::openapi::OpenApi {
    OPENAPI_DOC.clone()
}

fn apply_gateway_openapi_overrides(value: &mut Value) {
    value["info"]["license"] = json!({
        "name": "Apache-2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html"
    });
    value["security"] = json!([{ "bearerAuth": [] }]);

    let components = value
        .as_object_mut()
        .expect("OpenAPI root should be an object")
        .entry("components")
        .or_insert_with(|| json!({}));
    components["securitySchemes"] = json!({
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "opaque",
            "description": "SIE bearer token. Mutating pool/config/admin routes require the configured admin token; other protected routes accept a normal gateway token."
        }
    });

    if let Some(create_pool) = value
        .get_mut("components")
        .and_then(|components| components.get_mut("schemas"))
        .and_then(|schemas| schemas.get_mut("CreatePoolRequest"))
    {
        create_pool["required"] = json!(["name"]);
        if let Some(name_schema) = create_pool
            .get_mut("properties")
            .and_then(|properties| properties.get_mut("name"))
        {
            name_schema["minLength"] = json!(1);
            name_schema["maxLength"] = json!(128);
            name_schema["pattern"] = json!("^(?!_[dD][eE][fF][aA][uU][lL][tT]$)[A-Za-z0-9_-]+$");
            name_schema["description"] = json!(
                "Pool name used in gpu=\"pool/machine_profile\" routing. Names are stored and routed in lowercase. Only ASCII letters, digits, '_' and '-' are allowed; '_default' is reserved."
            );
        }
        if let Some(queue_pool_schema) = create_pool
            .get_mut("properties")
            .and_then(|properties| properties.get_mut("queue_pool"))
        {
            queue_pool_schema["minLength"] = json!(1);
            queue_pool_schema["maxLength"] = json!(128);
            queue_pool_schema["pattern"] =
                json!("^(?!_[dD][eE][fF][aA][uU][lL][tT]$)[A-Za-z0-9_-]+$");
        }
        create_pool["anyOf"] = json!([
            {
                "required": ["gpus"],
                "properties": {
                    "gpus": {
                        "minProperties": 1
                    }
                }
            },
            {
                "required": ["gpu_caps"],
                "properties": {
                    "gpu_caps": {
                        "minProperties": 1
                    }
                }
            }
        ]);
    }

    if let Some(queue_pool_schema) = value
        .get_mut("components")
        .and_then(|components| components.get_mut("schemas"))
        .and_then(|schemas| schemas.get_mut("PoolSpec"))
        .and_then(|pool_spec| pool_spec.get_mut("properties"))
        .and_then(|properties| properties.get_mut("queue_pool"))
    {
        queue_pool_schema["minLength"] = json!(1);
        queue_pool_schema["maxLength"] = json!(128);
        queue_pool_schema["pattern"] = json!("^(?!_[dD][eE][fF][aA][uU][lL][tT]$)[A-Za-z0-9_-]+$");
    }

    patch_chat_completion_schema(value);
    patch_queue_request_batch_limits(value);
    patch_rerank_schemas(value);

    let paths = value["paths"]
        .as_object_mut()
        .expect("OpenAPI paths should be an object");

    for path in ["/healthz", "/readyz", "/openapi.json", "/docs"] {
        if let Some(operation) = paths
            .get_mut(path)
            .and_then(|path_item| path_item.get_mut("get"))
        {
            operation["security"] = json!([]);
        }
    }

    if let Some(response) = paths
        .get_mut("/openapi.json")
        .and_then(|path_item| path_item.get_mut("get"))
        .and_then(|operation| operation.get_mut("responses"))
        .and_then(|responses| responses.get_mut("200"))
    {
        response["content"] = json!({
            "application/json": {
                "schema": {
                    "type": "object"
                }
            }
        });
    }

    inject_inference_msgpack_content(paths);
    inject_inference_response_headers(paths);
    inject_bearer_auth_error_responses(paths);
    annotate_slash_bearing_path_parameters(paths);
}

fn patch_queue_request_batch_limits(value: &mut Value) {
    let Some(schemas) = value
        .get_mut("components")
        .and_then(|components| components.get_mut("schemas"))
        .and_then(Value::as_object_mut)
    else {
        return;
    };
    let queue_maximum = json!(crate::queue::publisher::MAX_QUEUE_REQUEST_ITEMS);

    for schema_name in ["EncodeRequest", "ExtractRequest"] {
        if let Some(items) = schemas
            .get_mut(schema_name)
            .and_then(|schema| schema.get_mut("properties"))
            .and_then(|properties| properties.get_mut("items"))
        {
            items["maxItems"] = queue_maximum.clone();
        }
    }

    if let Some(items) = schemas
        .get_mut("ScoreRequest")
        .and_then(|schema| schema.get_mut("properties"))
        .and_then(|properties| properties.get_mut("items"))
    {
        items["maxItems"] = json!(crate::handlers::proxy::MAX_SCORE_ITEMS);
    }

    if let Some(variants) = schemas
        .get_mut("OpenAIEmbeddingInput")
        .and_then(|schema| schema.get_mut("oneOf"))
        .and_then(Value::as_array_mut)
    {
        for variant in variants {
            if variant.get("type").and_then(Value::as_str) == Some("array") {
                variant["maxItems"] = queue_maximum.clone();
            }
        }
    }
}

fn patch_chat_completion_schema(value: &mut Value) {
    patch_generate_request_schema(value);
    patch_generate_path(value);
    patch_chat_request_schema(value);
    patch_chat_message_schema(value);
    patch_completions_request_schema(value);
    patch_responses_request_schema(value);
    patch_chat_completion_paths(value);
    patch_completions_path(value);
    patch_responses_path(value);
}

fn patch_generate_request_schema(value: &mut Value) {
    let Some(properties) = value
        .get_mut("components")
        .and_then(|components| components.get_mut("schemas"))
        .and_then(|schemas| schemas.get_mut("GenerateRequest"))
        .and_then(|schema| schema.get_mut("properties"))
        .and_then(|properties| properties.as_object_mut())
    else {
        return;
    };
    properties.insert(
        "seed".to_string(),
        json!({
            "description": "Optional signed 64-bit per-request sampling seed. \
                            Reproducibility is best effort, not guaranteed, and depends on the \
                            active generation backend and deployment configuration. Non-integer \
                            or out-of-range values reject with 400 invalid_request.",
            "type": ["integer", "null"],
            "format": "int64",
            "minimum": i64::MIN,
            "maximum": i64::MAX,
        }),
    );
    properties.insert(
        "stream".to_string(),
        json!({
            "description": "SSE streaming. When true, the 200 response uses `text/event-stream`; \
                            each `data:` event contains a SIE-native `GenerateChunk` JSON object \
                            and the stream terminates with `data: [DONE]`. Defaults to false. \
                            Non-boolean values reject with 400 invalid_request.",
            "type": ["boolean", "null"],
        }),
    );
}

fn patch_generate_path(value: &mut Value) {
    let Some(post) = value
        .get_mut("paths")
        .and_then(|paths| paths.get_mut("/v1/generate/{model}"))
        .and_then(|path_item| path_item.get_mut("post"))
    else {
        return;
    };
    let Some(post) = post.as_object_mut() else {
        return;
    };

    post.insert(
        "description".to_string(),
        json!(
            "SIE-native generation with optional bounded inline images. Omit `stream` or set it \
             to false for a blocking JSON response; set `stream: true` for SIE-native \
             Server-Sent Events terminated by \
             `data: [DONE]`. The model path parameter must use the SIE-safe ID (for example \
             `Qwen__Qwen3-4B-Instruct`); HF-style slashes reject with 400."
        ),
    );

    let Some(response) = post
        .get_mut("responses")
        .and_then(|responses| responses.get_mut("200"))
        .and_then(|response| response.as_object_mut())
    else {
        return;
    };
    response.insert(
        "description".to_string(),
        json!(
            "Generated text as blocking JSON, or SIE-native SSE `GenerateChunk` events when \
             `stream: true`; SSE terminates with `data: [DONE]`."
        ),
    );
    let content = response
        .entry("content".to_string())
        .or_insert_with(|| json!({}));
    let Some(content) = content.as_object_mut() else {
        return;
    };
    content.insert(
        "text/event-stream".to_string(),
        json!({
            "schema": {
                "type": "string",
                "description": "SSE-framed text containing SIE-native `GenerateChunk` JSON in each `data:` event; the stream terminates with `data: [DONE]`."
            },
             "x-sie-event-schema": {
                 "$ref": "#/components/schemas/GenerateChunk"
             },
         }),
    );
}

fn patch_chat_request_schema(value: &mut Value) {
    let Some(properties) = value
        .get_mut("components")
        .and_then(|components| components.get_mut("schemas"))
        .and_then(|schemas| schemas.get_mut("ChatCompletionRequest"))
        .and_then(|schema| schema.get_mut("properties"))
        .and_then(|properties| properties.as_object_mut())
    else {
        return;
    };

    // Per-choice streaming: n > 1 with stream:true is now
    // supported (per-choice_index delta + per-choice closure chunks ride
    // before the single global [DONE]).
    properties.insert(
        "n".to_string(),
        json!({
            "description": "Number of candidate completions in [1, 128]. n>1 returns a \
                            multi-entry choices array; streaming with n>1 is supported \
                            (per-choice_index delta chunks + per-choice closure chunks \
                            ride before the single global [DONE]).",
            "type": "integer",
            "format": "int32",
            "minimum": 1,
            "maximum": 128,
            "nullable": true,
        }),
    );
    // best_of: range [1, 128]; cross-field rule best_of >= n; rejected with
    // 400 unsupported_field when stream:true (mirrors OpenAI). Cross-field
    // rules don't have a clean OpenAPI 3.0 expression — documented in prose.
    properties.insert(
        "best_of".to_string(),
        json!({
            "description": "Generate this many candidates and return the top `n` by \
                            cumulative logprob. Integer in [1, 128]. Cross-field rule: \
                            `best_of >= n` (otherwise 400 invalid_request). Rejected \
                            with 400 unsupported_field when `stream: true` (mirrors OpenAI).",
            "type": ["integer", "null"],
            "format": "int32",
            "minimum": 1,
            "maximum": 128,
        }),
    );
    // top_logprobs: [0, 20]; requires logprobs:true when > 0.
    properties.insert(
        "top_logprobs".to_string(),
        json!({
            "description": "Number of alternative top tokens to return alongside each \
                            chosen token's logprob. Integer in [0, 20]. Requires \
                            `logprobs: true` when > 0 (OpenAI rule; 400 invalid_request \
                            otherwise).",
            "type": ["integer", "null"],
            "format": "int32",
            "minimum": 0,
            "maximum": 20,
        }),
    );
    // logprobs: explicit boolean (not nullable in OpenAI's spec but optional).
    properties.insert(
        "logprobs".to_string(),
        json!({
            "description": "Return per-token logprobs. Boolean. When true, the chosen \
                            token's logprob (and optionally a top-N list via \
                            `top_logprobs`) rides on each `choices[].logprobs` entry.",
            "type": ["boolean", "null"],
        }),
    );
    // logit_bias: {token_id_string -> float in [-100, 100]}; capped at 1024 keys.
    properties.insert(
        "logit_bias".to_string(),
        json!({
            "description": "OpenAI `logit_bias` — `{token_id_string: bias_float}`. Keys \
                            must parse as integer token ids; values must be finite numbers \
                            in [-100.0, 100.0]. Map size capped at 1024 keys (request \
                            rejects with 400 invalid_request beyond the cap).",
            "type": ["object", "null"],
            "additionalProperties": {
                "type": "number",
                "format": "double",
                "minimum": -100.0,
                "maximum": 100.0,
            },
            "maxProperties": 1024,
        }),
    );
    // top_k: integer >= 1.
    properties.insert(
        "top_k".to_string(),
        json!({
            "description": "Non-OpenAI `top_k` (Together / Fireworks / vLLM extension): \
                            integer >= 1. Absent → top-k disabled (model default).",
            "type": ["integer", "null"],
            "format": "int32",
            "minimum": 1,
        }),
    );
    // repetition_penalty: float in (0.0, 2.0]; exclusiveMinimum is OpenAPI 3.0-compatible.
    properties.insert(
        "repetition_penalty".to_string(),
        json!({
            "description": "Non-OpenAI `repetition_penalty`: float in (0.0, 2.0] \
                            (1.0 = no penalty). Absent → sampler default.",
            "type": ["number", "null"],
            "format": "float",
            "exclusiveMinimum": 0.0,
            "maximum": 2.0,
        }),
    );
    // seed: signed 64-bit integer, forwarded unchanged.
    properties.insert(
        "seed".to_string(),
        json!({
            "description": "Optional signed 64-bit per-request sampling seed. \
                            Reproducibility is best effort, not guaranteed, and depends on the \
                            active generation backend and deployment configuration. Non-integer \
                            or out-of-range values reject with 400 invalid_request.",
            "type": ["integer", "null"],
            "format": "int64",
            "minimum": i64::MIN,
            "maximum": i64::MAX,
        }),
    );
    // stream: boolean; per-choice streaming for n>1 is supported.
    properties.insert(
        "stream".to_string(),
        json!({
            "description": "SSE streaming. When true, the response is a stream of \
                            `chat.completion.chunk` events terminated by `data: [DONE]`. \
                            For n > 1: per-`choice_index` delta chunks include a per-choice \
                            `delta.role:\"assistant\"` once per choice; per-choice closure \
                            chunks carry the `finish_reason` for that choice before the \
                            global `[DONE]`. Non-boolean values reject with 400 invalid_request.",
            "type": ["boolean", "null"],
        }),
    );
    // stream_options.include_usage is the only accepted sub-key.
    properties.insert(
        "stream_options".to_string(),
        json!({
            "description": "OpenAI `stream_options`. Accepted sub-key: `include_usage` \
                            (boolean — when true, the gateway emits a terminal `usage` frame \
                            before `[DONE]`). Any other sub-key rejects with 400 \
                            unsupported_field. Legal with `stream:false` (options ignored).",
            "type": ["object", "null"],
            "properties": {
                "include_usage": {
                    "type": ["boolean", "null"],
                    "description": "Emit a terminal `usage` frame before `[DONE]`.",
                }
            },
            "additionalProperties": false,
        }),
    );
    // tools: array of {type:"function", function:{name, parameters, description?}}.
    properties.insert(
        "tools".to_string(),
        json!({
            "description": "OpenAI tool-calling. Array of tool specs; each tool must match \
                            `{type:\"function\", function:{name, parameters, description?}}`. \
                            With n > 1, per-candidate `tool_calls` surface on \
                            `choices[i].message.tool_calls` (non-streaming) or ride on each \
                            `choices[].delta` (streaming).",
            "type": ["array", "null"],
            "items": {
                "type": "object",
                "required": ["type", "function"],
                "properties": {
                    "type": {"type": "string", "enum": ["function"]},
                    "function": {
                        "type": "object",
                        "required": ["name"],
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "parameters": {"type": "object"},
                        },
                    },
                },
            },
        }),
    );
    // tool_choice: one of "auto"|"none"|"required" or {type:"function", function:{name}}.
    properties.insert(
        "tool_choice".to_string(),
        json!({
            "description": "OpenAI `tool_choice`. One of: `\"auto\"`, `\"none\"`, \
                            `\"required\"`, or `{type:\"function\", function:{name}}` \
                            (named function). Requires `tools` to be set (otherwise 400 \
                            invalid_request). `\"required\"` and named-function choices \
                            cannot be combined with `response_format` (two competing \
                            grammars; 400 invalid_request).",
            "oneOf": [
                {"type": "string", "enum": ["auto", "none", "required"]},
                {
                    "type": "object",
                    "required": ["type", "function"],
                    "properties": {
                        "type": {"type": "string", "enum": ["function"]},
                        "function": {
                            "type": "object",
                            "required": ["name"],
                            "properties": {"name": {"type": "string"}},
                        },
                    },
                },
            ],
        }),
    );
    properties.insert(
        "parallel_tool_calls".to_string(),
        json!({
            "description": "OpenAI `parallel_tool_calls` — boolean controlling whether the \
                            model may emit multiple tool calls per turn.",
            "type": ["boolean", "null"],
        }),
    );
    // response_format: {type:"text"|"json_object"|"json_schema", json_schema?:{...}}.
    properties.insert(
        "response_format".to_string(),
        json!({
            "description": "OpenAI `response_format` — translated into a grammar spec on \
                            the worker. Accepted shapes: `{type:\"text\"}`, \
                            `{type:\"json_object\"}`, `{type:\"json_schema\", \
                            json_schema:{...}}`. Cannot be combined with a forcing \
                            `tool_choice` (`\"required\"` or a named function) — two \
                            competing grammars on one request reject with 400 invalid_request.",
            "type": ["object", "null"],
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["text", "json_object", "json_schema"],
                },
                "json_schema": {"type": "object"},
            },
        }),
    );
    properties.insert(
        "stop".to_string(),
        json!({
            "description": "Either a string or an array of strings, mirroring OpenAI.",
            "oneOf": [
                {"type": "string"},
                {"type": "array", "items": {"type": "string"}}
            ]
        }),
    );
    // lora_adapter: top-level /generate lora_path kwarg (the worker forwards to
    // SGLang where it ends up in sampling_params.lora_path — the wire shape at
    // the gateway→worker boundary is a top-level kwarg, not a sampling_params field).
    properties.insert(
        "lora_adapter".to_string(),
        json!({
            "description": "SIE extension: non-empty served-name of a LoRA adapter declared \
                            in the model profile's `lora_paths`. Absent → the base model. \
                            Unknown name → 400 with `param:\"lora_adapter\"`. The gateway \
                            forwards it to the worker as a top-level `lora_path` generation \
                            kwarg (SGLang then selects the adapter by served name; the \
                            sampling-params placement is an SGLang implementation detail, \
                            not part of the SIE wire contract).",
            "type": ["string", "null"],
            "minLength": 1,
        }),
    );
    properties.insert(
        "user".to_string(),
        json!({
            "description": "OpenAI `user` — Sensitive PII. Accepted-and-dropped: \
                            debug-logged only, never persisted, never forwarded to \
                            the worker.",
            "type": ["string", "null"],
            "x-sensitive": true
        }),
    );
}

fn patch_chat_message_schema(value: &mut Value) {
    let Some(schema) = value
        .get_mut("components")
        .and_then(|components| components.get_mut("schemas"))
        .and_then(|schemas| schemas.get_mut("ChatCompletionMessage"))
    else {
        return;
    };
    let Some(properties) = schema
        .get_mut("properties")
        .and_then(|properties| properties.as_object_mut())
    else {
        return;
    };
    // content: string OR array of text-only content parts; null on
    // assistant-with-tool_calls messages.
    properties.insert(
        "content".to_string(),
        json!({
            "description": "Either a string or an array of content parts \
                            (`{type:\"text\"|\"input_text\", text:\"...\"}`). Image parts \
                            (`image_url` / `input_image`) carrying a base64 `data:` URI are \
                            accepted for generation models that declare `inputs.image`; non-vision models \
                            reject with 400 unsupported_field and remote (non-`data:`) URLs \
                            reject with 400 invalid_request. May be \
                            `null` on a `role:\"assistant\"` message that carries `tool_calls`.",
            "oneOf": [
                {"type": "string"},
                {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["type"],
                        "properties": {
                            "type": {"type": "string", "enum": ["text", "input_text", "image_url", "input_image"]},
                            "text": {"type": "string"},
                            "image_url": {
                                "description": "Image payload for `image_url` / `input_image` parts: a base64 `data:` URI, either as a bare string or as `{ \"url\": \"data:...\" }`. Remote (non-`data:`) URLs reject with 400 invalid_request.",
                                "oneOf": [
                                    {"type": "string"},
                                    {"type": "object", "properties": {"url": {"type": "string"}}},
                                ],
                            },
                        },
                    },
                },
                {"type": "null"},
            ],
        }),
    );
    // tool_calls: array of {id, type:"function", function:{name, arguments}}.
    properties.insert(
        "tool_calls".to_string(),
        json!({
            "description": "OpenAI tool-call replay on `role:\"assistant\"` messages. Each \
                            entry MUST match `{id, type:\"function\", function:{name, \
                            arguments}}`; `arguments` is a JSON-encoded string. Rejected on \
                            other roles with 400.",
            "type": ["array", "null"],
            "items": {
                "type": "object",
                "required": ["id", "type", "function"],
                "properties": {
                    "id": {"type": "string"},
                    "type": {"type": "string", "enum": ["function"]},
                    "function": {
                        "type": "object",
                        "required": ["name", "arguments"],
                        "properties": {
                            "name": {"type": "string"},
                            "arguments": {
                                "type": "string",
                                "description": "JSON-encoded argument string (OpenAI convention).",
                            },
                        },
                    },
                },
            },
        }),
    );
    properties.insert(
        "tool_call_id".to_string(),
        json!({
            "description": "Required on `role:\"tool\"` messages (matches the assistant turn's \
                            `tool_calls[].id`). Rejected on every other role with 400 \
                            invalid_request.",
            "type": ["string", "null"],
        }),
    );
    // ``content`` is required EXCEPT on assistant-with-tool_calls (per parser),
    // which OpenAPI 3.0 can't express precisely without conditional schemas.
    // Document the optionality in prose; keep ``role`` required.
    if let Some(required) = schema.get_mut("required").and_then(|r| r.as_array_mut()) {
        required.retain(|v| v.as_str() != Some("content"));
    }
}

fn patch_completions_request_schema(value: &mut Value) {
    let Some(schema) = value
        .get_mut("components")
        .and_then(|components| components.get_mut("schemas"))
        .and_then(|schemas| schemas.get_mut("CompletionsRequest"))
    else {
        return;
    };
    let Some(properties) = schema
        .get_mut("properties")
        .and_then(|properties| properties.as_object_mut())
    else {
        return;
    };
    properties.insert(
        "stop".to_string(),
        json!({
            "description": "Either a string or an array of strings, mirroring OpenAI.",
            "oneOf": [
                {"type": "string"},
                {"type": "array", "items": {"type": "string"}}
            ]
        }),
    );
    properties.insert(
        "n".to_string(),
        json!({
            "description": "Single-candidate only: integer `1` (or absent) accepted; `n > 1` \
                            rejects with 400 unsupported_field (use chat for multi-candidate). \
                            `n == 0` rejects with 400 invalid_request.",
            "type": ["integer", "null"],
            "format": "int32",
            "minimum": 1,
            "maximum": 1,
        }),
    );
    properties.insert(
        "stream".to_string(),
        json!({
            "description": "SSE streaming. When true, the response is a stream of \
                            `text_completion` events terminated by `data: [DONE]`. \
                            Non-boolean values reject with 400 invalid_request.",
            "type": ["boolean", "null"],
        }),
    );
    properties.insert(
        "seed".to_string(),
        json!({
            "description": "Optional signed 64-bit per-request sampling seed. \
                            Reproducibility is best effort, not guaranteed, and depends on the \
                            active generation backend and deployment configuration. Non-integer \
                            or out-of-range values reject with 400 invalid_request.",
            "type": ["integer", "null"],
            "format": "int64",
            "minimum": i64::MIN,
            "maximum": i64::MAX,
        }),
    );
}

fn patch_responses_request_schema(value: &mut Value) {
    let Some(schema) = value
        .get_mut("components")
        .and_then(|components| components.get_mut("schemas"))
        .and_then(|schemas| schemas.get_mut("ResponsesRequest"))
    else {
        return;
    };
    let Some(properties) = schema
        .get_mut("properties")
        .and_then(|properties| properties.as_object_mut())
    else {
        return;
    };
    // input: either a string OR an array of {role, content} messages.
    properties.insert(
        "input".to_string(),
        json!({
            "description": "Either a string prompt OR an array of `{role, content}` messages \
                            (array-input support). Array form: `role` is one of \
                            `\"system\" | \"user\" | \"assistant\" | \"developer\"` (\"developer\" \
                            normalizes to \"system\"). `content` is a string or an array of \
                            text-only content parts; image parts (`image_url` / `input_image`) \
                            reject with 400 unsupported_field. The array must not be empty.",
            "oneOf": [
                {"type": "string"},
                {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["role", "content"],
                        "properties": {
                            "role": {
                                "type": "string",
                                "enum": ["system", "user", "assistant", "developer"],
                            },
                            "content": {
                                "oneOf": [
                                    {"type": "string"},
                                    {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "required": ["type"],
                                            "properties": {
                                                "type": {
                                                    "type": "string",
                                                    "enum": ["text", "input_text"],
                                                },
                                                "text": {"type": "string"},
                                            },
                                        },
                                    },
                                ],
                            },
                        },
                    },
                },
            ],
        }),
    );
    properties.insert(
        "stream".to_string(),
        json!({
            "description": "Accepted only as `false` (or absent). `true` rejects with 400 \
                            unsupported_field because Responses SSE is not supported yet.",
            "type": ["boolean", "null"],
            "enum": [false, null],
        }),
    );
    properties.insert(
        "seed".to_string(),
        json!({
            "description": "Optional signed 64-bit per-request sampling seed. \
                            Reproducibility is best effort, not guaranteed, and depends on the \
                            active generation backend and deployment configuration. Non-integer \
                            or out-of-range values reject with 400 invalid_request.",
            "type": ["integer", "null"],
            "format": "int64",
            "minimum": i64::MIN,
            "maximum": i64::MAX,
        }),
    );
}

fn patch_chat_completion_paths(value: &mut Value) {
    let Some(op) = value
        .get_mut("paths")
        .and_then(|paths| paths.get_mut("/v1/chat/completions"))
        .and_then(|path_item| path_item.get_mut("post"))
    else {
        return;
    };
    if let Some(op_obj) = op.as_object_mut() {
        op_obj.insert(
            "description".to_string(),
            json!(
                "OpenAI-compatible chat completions. Strict allow-list parser: \
                 unknown top-level fields reject with 400 \
                 `unsupported_field`. Streaming (`stream: true`) is supported and emits \
                 SSE `chat.completion.chunk` events; `n > 1` streaming fans candidates \
                 out as per-`choice_index` delta chunks with per-choice closure chunks \
                 (each carrying `finish_reason`) before the single global `[DONE]`. \
                 Content is either a string or an array of content parts; image parts \
                 (`image_url` / `input_image`) with a base64 `data:` URI are accepted for \
                 generation models declaring `inputs.image` (non-vision/encode-only models reject 400 \
                 `unsupported_field`; remote URLs reject 400 `invalid_request`). \
                 `lora_adapter` is forwarded to the worker as a \
                 top-level `lora_path` generation kwarg."
            ),
        );
    }
}

fn patch_completions_path(value: &mut Value) {
    let Some(post) = value
        .get_mut("paths")
        .and_then(|paths| paths.get_mut("/v1/completions"))
        .and_then(|path_item| path_item.get_mut("post"))
    else {
        return;
    };
    if let Some(op_obj) = post.as_object_mut() {
        op_obj.insert(
            "description".to_string(),
            json!(
                "OpenAI-compatible legacy Completions. Strict allow-list parser: \
                 unknown top-level fields reject with 400 \
                 `unsupported_field`. `stream: true` is supported (SSE `text_completion`). \
                 Known-rejected fields: `echo`, `suffix`, `logprobs`, `best_of`, `n > 1`, \
                 batched array `prompt` — each rejects with 400 `unsupported_field`. The \
                 response body no longer carries the always-null `logprobs` field."
            ),
        );
        op_obj.insert(
            "summary".to_string(),
            json!("`/v1/completions` — legacy OpenAI Completions (single-candidate, raw-prompt)."),
        );
        // Document the request body shape (utoipa generates this from the
        // attribute on the handler — but proxy.rs has no request_body
        // attribute on proxy_completions, so we add the schema reference here).
        let request_body = op_obj
            .entry("requestBody".to_string())
            .or_insert_with(|| json!({}));
        if let Some(request_body) = request_body.as_object_mut() {
            request_body.insert("required".to_string(), json!(true));
            request_body.insert(
                "content".to_string(),
                json!({
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/CompletionsRequest"}
                    }
                }),
            );
        }
    }
}

fn patch_responses_path(value: &mut Value) {
    let Some(post) = value
        .get_mut("paths")
        .and_then(|paths| paths.get_mut("/v1/responses"))
        .and_then(|path_item| path_item.get_mut("post"))
    else {
        return;
    };
    if let Some(op_obj) = post.as_object_mut() {
        op_obj.insert(
            "description".to_string(),
            json!(
                "OpenAI Responses API (MVP). Strict allow-list parser: \
                 unknown top-level fields reject with 400 \
                 `unsupported_field`. `input` is either a string prompt OR an array of \
                 `{role, content}` messages. \
                 Known-rejected fields: `tools`, `tool_choice`, `previous_response_id`, \
                 `reasoning`, `background`, `metadata`, `instructions` — each rejects \
                 with 400 `unsupported_field`. `stream: true` is rejected (Responses SSE \
                 is deferred). Multimodal image content parts reject with 400 \
                 `unsupported_field` on the array form."
            ),
        );
        op_obj.insert(
            "summary".to_string(),
            json!("`/v1/responses` — OpenAI Responses API (MVP, stateless single-turn)."),
        );
        let request_body = op_obj
            .entry("requestBody".to_string())
            .or_insert_with(|| json!({}));
        if let Some(request_body) = request_body.as_object_mut() {
            request_body.insert("required".to_string(), json!(true));
            request_body.insert(
                "content".to_string(),
                json!({
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ResponsesRequest"}
                    }
                }),
            );
        }
    }
}

/// Duplicate ``application/json`` schemas to the msgpack media types accepted by queue inference.
fn inject_inference_msgpack_content(paths: &mut serde_json::Map<String, Value>) {
    for path in [
        "/v1/encode/{model}",
        "/v1/score/{model}",
        "/v1/extract/{model}",
    ] {
        let Some(path_item) = paths.get_mut(path) else {
            continue;
        };
        let Some(post) = path_item.get_mut("post").and_then(|p| p.as_object_mut()) else {
            continue;
        };
        if let Some(rb) = post.get_mut("requestBody").and_then(|r| r.as_object_mut()) {
            if let Some(content) = rb.get_mut("content").and_then(|c| c.as_object_mut()) {
                if let Some(json_entry) = content.get("application/json").cloned() {
                    for media_type in [
                        "application/msgpack",
                        "application/x-msgpack",
                        "application/vnd.msgpack",
                    ] {
                        content.insert(media_type.to_string(), json_entry.clone());
                    }
                }
            }
        }
        let Some(responses) = post.get_mut("responses").and_then(|r| r.as_object_mut()) else {
            continue;
        };
        let Some(ok) = responses.get_mut("200").and_then(|x| x.as_object_mut()) else {
            continue;
        };
        let Some(content) = ok.get_mut("content").and_then(|c| c.as_object_mut()) else {
            continue;
        };
        if let Some(json_entry) = content.get("application/json").cloned() {
            for media_type in [
                "application/msgpack",
                "application/x-msgpack",
                "application/vnd.msgpack",
            ] {
                content.insert(media_type.to_string(), json_entry.clone());
            }
        }
    }
}

fn header(description: &str) -> Value {
    json!({
        "description": description,
        "schema": { "type": "string" }
    })
}

fn sha256_header(description: &str) -> Value {
    json!({
        "description": description,
        "schema": {
            "type": "string",
            "pattern": "^[0-9a-f]{64}$"
        }
    })
}

fn merge_headers(response: &mut Value, headers: &Value) {
    let Some(response) = response.as_object_mut() else {
        return;
    };
    let Some(header_map) = headers.as_object() else {
        return;
    };
    let headers = response
        .entry("headers")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .expect("OpenAPI response headers should be an object");
    for (name, value) in header_map {
        headers.entry(name.clone()).or_insert(value.clone());
    }
}

fn inject_inference_response_headers(paths: &mut serde_json::Map<String, Value>) {
    let queue_success_headers = json!({
        "X-SIE-Version": header("Gateway package version that handled the request"),
        "X-SIE-Server-Version": header("Gateway-compatible server version advertised by this gateway"),
        "X-SIE-Model-Revision": header(
            "Immutable deployed bundle/config execution revision that handled the request, when available",
        ),
        "X-SIE-Execution-Identity-SHA256": sha256_header(
            "Worker-origin SHA-256 identity of the immutable release and realized serving resources, when available",
        ),
        "X-SIE-Request-Id": header("Gateway request id for queue-backed inference"),
        "X-SIE-Worker": header("Logical queue worker tag that produced the response"),
        "X-Queue-Publish-Time": header("Milliseconds spent publishing work to the queue"),
        "X-Queue-Wait-Time": header("Milliseconds spent waiting for worker results"),
        "X-Queue-Time": header("Worker-reported queue time in milliseconds"),
        "X-Inference-Time": header("Worker-reported inference time in milliseconds"),
        "X-Tokenization-Time": header("Worker-reported tokenization time in milliseconds, when available"),
        "X-Postprocessing-Time": header("Worker-reported postprocessing time in milliseconds, when available"),
        "X-Payload-Fetch-Time": header("Worker-reported offloaded payload fetch time in milliseconds, when available")
    });
    let retryable_error_headers = json!({
        "Retry-After": header("Suggested retry delay in seconds, when retryable"),
        "X-SIE-Error-Code": header("SDK-stable gateway or worker error code"),
        "X-SIE-Version": header("Gateway package version that handled the request"),
        "X-SIE-Server-Version": header("Gateway-compatible server version advertised by this gateway")
    });
    let terminal_error_headers = json!({
        "X-SIE-Error-Code": header("SDK-stable gateway or worker error code"),
        "X-SIE-Error-Version": header("Gateway package version associated with the error envelope"),
        "X-SIE-Version": header("Gateway package version that handled the request"),
        "X-SIE-Server-Version": header("Gateway-compatible server version advertised by this gateway")
    });

    for path in [
        "/v1/encode/{model}",
        "/v1/score/{model}",
        "/v1/extract/{model}",
        "/v1/generate/{model}",
        "/v1/audio/transcriptions",
    ] {
        let Some(responses) = paths
            .get_mut(path)
            .and_then(|path_item| path_item.get_mut("post"))
            .and_then(|post| post.get_mut("responses"))
            .and_then(|responses| responses.as_object_mut())
        else {
            continue;
        };
        if let Some(response) = responses.get_mut("200") {
            merge_headers(response, &queue_success_headers);
        }
        if let Some(response) = responses.get_mut("502") {
            merge_headers(response, &terminal_error_headers);
        }
        if let Some(response) = responses.get_mut("503") {
            merge_headers(response, &retryable_error_headers);
        }
    }

    if let Some(responses) = paths
        .get_mut("/v1/embeddings")
        .and_then(|path_item| path_item.get_mut("post"))
        .and_then(|post| post.get_mut("responses"))
        .and_then(|responses| responses.as_object_mut())
    {
        if let Some(response) = responses.get_mut("200") {
            merge_headers(response, &queue_success_headers);
        }
        if let Some(response) = responses.get_mut("502") {
            merge_headers(response, &terminal_error_headers);
        }
        if let Some(response) = responses.get_mut("503") {
            merge_headers(response, &retryable_error_headers);
        }
    }

    for path in ["/v1/chat/completions", "/v1/completions", "/v1/responses"] {
        let Some(responses) = paths
            .get_mut(path)
            .and_then(|path_item| path_item.get_mut("post"))
            .and_then(|post| post.get_mut("responses"))
            .and_then(|responses| responses.as_object_mut())
        else {
            continue;
        };
        if let Some(response) = responses.get_mut("200") {
            merge_headers(response, &queue_success_headers);
        }
        if let Some(response) = responses.get_mut("502") {
            merge_headers(response, &terminal_error_headers);
        }
        if let Some(response) = responses.get_mut("503") {
            merge_headers(response, &retryable_error_headers);
        }
    }
}

/// Add documented auth responses for bearer-protected operations (spec completeness).
fn inject_bearer_auth_error_responses(paths: &mut serde_json::Map<String, Value>) {
    let unauthorized = json!({
        "description": "Missing or invalid bearer token (inference token)",
        "content": {
            "application/json": {
                "schema": { "$ref": "#/components/schemas/StandardApiError" }
            }
        }
    });
    let forbidden = json!({
        "description": "Valid bearer token but admin token required for this mutation (or admin token not configured)",
        "content": {
            "application/json": {
                "schema": { "$ref": "#/components/schemas/StandardApiError" }
            }
        }
    });
    let auth_misconfigured = json!({
        "description": "Gateway auth enabled but no tokens configured",
        "content": {
            "application/json": {
                "schema": { "$ref": "#/components/schemas/StandardApiError" }
            }
        }
    });

    for (path, path_item) in paths.iter_mut() {
        let Some(path_item) = path_item.as_object_mut() else {
            continue;
        };
        for (method, op) in path_item.iter_mut() {
            if !matches!(method.as_str(), "get" | "post" | "put" | "delete" | "patch") {
                continue;
            }
            let Some(op_ob) = op.as_object_mut() else {
                continue;
            };
            let secured = op_ob
                .get("security")
                .and_then(|s| s.as_array())
                .map(|a| !a.is_empty())
                .unwrap_or(true);
            if !secured {
                continue;
            }
            let Some(responses) = op_ob.get_mut("responses").and_then(|r| r.as_object_mut()) else {
                continue;
            };
            responses
                .entry("401".to_string())
                .or_insert(unauthorized.clone());
            merge_auth_misconfigured_response(responses, &auth_misconfigured);

            if admin_mutation(method.as_str(), path) {
                responses
                    .entry("403".to_string())
                    .or_insert(forbidden.clone());
            }
        }
    }
}

fn standard_api_error_schema() -> Value {
    json!({ "$ref": "#/components/schemas/StandardApiError" })
}

fn schema_allows_standard_api_error(schema: &Value) -> bool {
    if schema == &standard_api_error_schema() {
        return true;
    }
    schema
        .get("oneOf")
        .and_then(|one_of| one_of.as_array())
        .map(|schemas| schemas.iter().any(schema_allows_standard_api_error))
        .unwrap_or(false)
}

fn merge_auth_misconfigured_response(
    responses: &mut serde_json::Map<String, Value>,
    auth_misconfigured: &Value,
) {
    let Some(existing) = responses.get_mut("500") else {
        responses.insert("500".to_string(), auth_misconfigured.clone());
        return;
    };

    let Some(existing) = existing.as_object_mut() else {
        return;
    };
    let description = existing
        .entry("description")
        .or_insert_with(|| json!("Internal server error"));
    if let Some(description_text) = description.as_str() {
        if !description_text.contains("auth enabled but no tokens configured") {
            *description = json!(format!(
                "{description_text}; gateway auth enabled but no tokens configured"
            ));
        }
    }

    let content = existing
        .entry("content")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .expect("OpenAPI response content should be an object");
    let json_content = content
        .entry("application/json")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .expect("OpenAPI application/json response should be an object");
    let schema = json_content.entry("schema").or_insert_with(|| json!(null));
    if schema.is_null() {
        *schema = standard_api_error_schema();
    } else if !schema_allows_standard_api_error(schema) {
        let old_schema = schema.clone();
        *schema = json!({
            "oneOf": [
                old_schema,
                standard_api_error_schema()
            ]
        });
    }
}

fn admin_mutation(method: &str, path: &str) -> bool {
    if !matches!(method, "post" | "put" | "delete") {
        return false;
    }
    path.starts_with("/v1/config") || path.starts_with("/v1/admin") || path.starts_with("/v1/pools")
}

fn annotate_slash_bearing_path_parameters(paths: &mut serde_json::Map<String, Value>) {
    for (path, param_name, runtime_path) in [
        ("/v1/models/{model}", "model", "/v1/models/{*model}"),
        ("/v1/encode/{model}", "model", "/v1/encode/{*model}"),
        ("/v1/score/{model}", "model", "/v1/score/{*model}"),
        ("/v1/extract/{model}", "model", "/v1/extract/{*model}"),
        ("/v1/generate/{model}", "model", "/v1/generate/{*model}"),
        ("/v1/configs/models/{id}", "id", "/v1/configs/models/{*id}"),
        (
            "/v1/configs/models/{id}/status",
            "id",
            "/v1/configs/models/{*id}",
        ),
    ] {
        let Some(path_item) = paths.get_mut(path).and_then(|item| item.as_object_mut()) else {
            continue;
        };
        for operation in path_item.values_mut() {
            let Some(operation) = operation.as_object_mut() else {
                continue;
            };
            operation.insert("x-sie-axum-catch-all".to_string(), json!(runtime_path));
            let Some(parameters) = operation
                .get_mut("parameters")
                .and_then(|parameters| parameters.as_array_mut())
            else {
                continue;
            };
            for parameter in parameters {
                if parameter.get("name").and_then(|name| name.as_str()) != Some(param_name) {
                    continue;
                }
                parameter["description"] = json!(format!(
                    "Model id. Runtime route is Axum catch-all `{runtime_path}`; clients using this OpenAPI path template should percent-encode slashes in model ids, for example `BAAI%2Fbge-m3`."
                ));
            }
        }
    }
}

fn patch_rerank_schemas(value: &mut Value) {
    let Some(schemas) = value
        .get_mut("components")
        .and_then(|components| components.get_mut("schemas"))
        .and_then(Value::as_object_mut)
    else {
        return;
    };

    for name in ["RerankRequest", "RerankV2Request"] {
        let Some(properties) = schemas
            .get_mut(name)
            .and_then(|schema| schema.get_mut("properties"))
            .and_then(Value::as_object_mut)
        else {
            continue;
        };
        for field in ["model", "query"] {
            properties[field]["minLength"] = json!(1);
            properties[field]["pattern"] = json!(r".*\S.*");
        }
        properties["documents"]["items"]["minLength"] = json!(1);
        properties["documents"]["items"]["pattern"] = json!(r".*\S.*");
    }

    if let Some(profile) = schemas
        .get_mut("RerankOptions")
        .and_then(|schema| schema.get_mut("properties"))
        .and_then(|properties| properties.get_mut("profile"))
    {
        profile["minLength"] = json!(1);
        profile["pattern"] = json!(r".*\S.*");
    }

    if let Some(properties) = schemas
        .get_mut("RerankV2Request")
        .and_then(|schema| schema.get_mut("properties"))
        .and_then(Value::as_object_mut)
    {
        for field in ["max_tokens_per_doc", "priority"] {
            properties[field] = json!({
                "type": "null",
                "description": format!(
                    "`{field}` is not supported by this SIE compatibility subset; omit it or send null."
                )
            });
        }
    }
}

pub fn write_openapi_json(path: Option<&Path>) -> Result<(), Box<dyn std::error::Error>> {
    match path {
        Some(path) => std::fs::write(path, OPENAPI_JSON.as_bytes())?,
        None => print!("{}", OPENAPI_JSON.as_str()),
    }
    Ok(())
}

#[utoipa::path(
    get,
    path = "/v1/configs/models/{id}/status",
    operation_id = "get_model_config_status",
    tag = "config",
    params(("id" = String, Path, description = "Model id; percent-encode slashes when using OpenAPI-generated clients")),
    responses(
        (status = 200, description = "Per-replica worker acknowledgement status", body = ModelConfigStatusResponse),
        (status = 404, description = "Model not found", body = StandardApiError)
    )
)]
#[allow(dead_code)]
pub fn get_model_config_status_doc() {}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ErrorDetailCore {
    pub code: String,
    pub message: String,
}

/// FastAPI-style error envelope for most gateway JSON errors.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct StandardApiError {
    pub detail: ErrorDetailCore,
}

/// ``detail`` when the caller's ``X-SIE-MACHINE-PROFILE`` GPU is not in the gateway allow-list.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct GpuNotConfiguredDetail {
    pub code: String,
    pub message: String,
    pub gpu: String,
    pub configured_gpu_types: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct GpuNotConfiguredError {
    pub detail: GpuNotConfiguredDetail,
}

/// SDK-style ``502`` body for ``MODEL_LOAD_FAILED`` (SDK short-circuit).
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct GatewayModelLoadFailedDetail {
    pub code: String,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_class: Option<String>,
    pub attempts: i32,
    pub permanent: bool,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct GatewayModelLoadFailedResponse {
    pub error: GatewayModelLoadFailedDetail,
}

/// OpenAI-compatible ``POST /v1/embeddings`` request (subset supported on gateway).
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct OpenAIEmbeddingRequest {
    pub model: String,
    pub input: OpenAIEmbeddingInput,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<OpenAIEmbeddingEncodingFormat>,
    /// Accepted but ignored; the gateway returns the model's native dimension.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    /// Accepted but ignored; kept for OpenAI SDK compatibility.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum OpenAIEmbeddingInput {
    String(String),
    Strings(Vec<String>),
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum OpenAIEmbeddingEncodingFormat {
    Float,
    Base64,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum OpenAIEmbeddingVector {
    Float(Vec<f64>),
    Base64(String),
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct OpenAIEmbeddingDataEntry {
    pub object: String,
    pub embedding: OpenAIEmbeddingVector,
    pub index: usize,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct OpenAIEmbeddingUsage {
    pub prompt_tokens: u64,
    pub total_tokens: u64,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct OpenAIEmbeddingsListResponse {
    pub object: String,
    pub data: Vec<OpenAIEmbeddingDataEntry>,
    pub model: String,
    pub usage: OpenAIEmbeddingUsage,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum OpenAITranscriptionResponseFormat {
    Json,
    Text,
    Srt,
    VerboseJson,
    Vtt,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum OpenAITranscriptionTimestampGranularity {
    Word,
    Segment,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct OpenAITranscriptionRequest {
    #[schema(value_type = String, format = Binary)]
    pub file: String,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<OpenAITranscriptionResponseFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(minimum = 0, maximum = 1)]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(
        default,
        rename = "timestamp_granularities[]",
        skip_serializing_if = "Option::is_none"
    )]
    #[schema(max_items = 2)]
    pub timestamp_granularities: Option<Vec<OpenAITranscriptionTimestampGranularity>>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct OpenAITranscriptionUsage {
    #[serde(rename = "type")]
    pub usage_type: String,
    pub seconds: f64,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct OpenAITranscriptionWord {
    pub word: String,
    pub start: f64,
    pub end: f64,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct OpenAITranscriptionSegment {
    pub id: u64,
    pub start: f64,
    pub end: f64,
    pub text: String,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct OpenAITranscriptionResponse {
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<OpenAITranscriptionUsage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub segments: Option<Vec<OpenAITranscriptionSegment>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub words: Option<Vec<OpenAITranscriptionWord>>,
}
// ── /v1/generate/{model} schemas ──────────────────────────────────

/// One inline image on the SIE-native generate surface.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct NativeGenerateImage {
    /// Canonical standard-base64 encoded image bytes, at most 16 MiB decoded.
    /// Remote URLs are not accepted.
    #[schema(min_length = 1, max_length = 22369624)]
    pub data: String,
    /// Short media-format hint such as ``png`` or ``jpeg``.
    #[schema(min_length = 1, max_length = 32, pattern = "^[A-Za-z0-9.+-]+$")]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
}

/// SIE-native text-generation request. Set ``stream`` to true for SSE.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct GenerateRequest {
    #[schema(min_length = 1)]
    pub prompt: String,
    /// Optional inline images paired with ``prompt``. When present, the worker
    /// renders one user turn through the model's native chat template.
    #[schema(min_items = 1, max_items = 16)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<NativeGenerateImage>>,
    #[schema(minimum = 1)]
    pub max_new_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Governed generation runtime options. Explicit top-level sampler fields
    /// override matching values in ``default_sampling``.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub options: Option<Value>,
    /// Stop sequences for the native generate surface.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    /// Optional signed 64-bit per-request sampling seed. Reproducibility is
    /// best effort and depends on the active backend and deployment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    /// Per-token additive sampler bias. Keys are integer token ids encoded as
    /// strings; values are finite numbers in ``[-100, 100]``. At most 1024
    /// entries are accepted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<std::collections::BTreeMap<String, f64>>,
    /// Return per-token log probabilities on native SSE chunks. Native
    /// blocking responses reject this field because they have no logprob
    /// response member.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    /// Number of alternative tokens per position. Requires ``logprobs: true``
    /// and ``stream: true``.
    #[schema(minimum = 0, maximum = 20)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,
    /// SSE streaming. Each ``data:`` event contains a SIE-native
    /// ``GenerateChunk`` JSON object; the stream ends with ``data: [DONE]``.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Optional grammar object accepted by the gateway grammar validator.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grammar: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    /// Served LoRA adapter name. The gateway validates it against the resolved
    /// model before queue dispatch.
    #[schema(min_length = 1)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora_adapter: Option<String>,
    /// Sensitive PII - parsed and dropped, never logged or forwarded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct GenerateUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct GenerateResponse {
    pub model: String,
    pub text: String,
    pub finish_reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<GenerateUsage>,
    pub attempt_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tpot_ms: Option<f64>,
}

/// Error carried by a terminal SIE-native generation SSE event.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct GenerateChunkError {
    pub code: String,
    pub message: String,
}

/// One JSON payload from a SIE-native generation SSE ``data:`` event.
/// The stream is terminated separately by the literal ``data: [DONE]``.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct GenerateChunk {
    pub request_id: String,
    pub seq: u32,
    pub text_delta: String,
    pub done: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<GenerateUsage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<f64>,
    /// Per-token log probabilities aligned with ``text_delta``. Present only
    /// when the streaming request set ``logprobs: true``.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Vec<Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<GenerateChunkError>,
}

// ── /v1/chat/completions schemas ──────────────────────────────────

/// One chat message in an OpenAI chat-completions request.
///
/// Per the strict allow-list parser, ``content`` is either a
/// plain string OR a list of content parts (``{type:"text"|"input_text",
/// text:"..."}``). ``image_url`` / ``input_image`` parts carrying a base64
/// ``data:`` URI are accepted for generation models declaring ``inputs.image`` (non-vision
/// models reject 400 ``unsupported_field``; remote URLs reject 400
/// ``invalid_request``). Assistant messages that carry ``tool_calls`` may set
/// ``content: null``.
///
/// ``tool_calls`` is accepted only on ``role:"assistant"`` messages; each
/// entry has the validated shape ``{id, type:"function", function:{name, arguments}}``
/// where ``arguments`` is a JSON-encoded string (OpenAI convention).
/// ``tool_call_id`` is required on ``role:"tool"`` messages and rejected
/// on every other role.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ChatCompletionMessage {
    /// One of ``"system" | "user" | "assistant" | "tool" | "developer"``.
    /// ``tool`` carries multi-turn tool-call replay; ``developer`` is OpenAI's
    /// newer alias for ``system`` and is normalized to ``system``. Any other
    /// role rejects with 400 ``invalid_request``.
    pub role: String,
    /// Either a string or an array of content parts. ``image_url`` /
    /// ``input_image`` parts with a base64 ``data:`` URI are accepted for
    /// vision-capable generation models (``inputs.image`` + a generate task); non-vision/encode-only models reject 400
    /// ``unsupported_field`` and remote URLs reject 400 ``invalid_request``.
    /// May be ``null`` on an assistant message that carries ``tool_calls``.
    pub content: Value,
    /// OpenAI tool-call replay on ``role:"assistant"`` messages.
    /// Each entry MUST match ``{id, type:"function", function:{name, arguments}}``;
    /// ``arguments`` is a JSON-encoded string. Rejected on other roles.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Value>,
    /// Required on ``role:"tool"`` messages (matches the assistant turn's
    /// ``tool_calls[].id``). Rejected on every other role with 400.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// OpenAI-compatible ``POST /v1/chat/completions`` request.
///
/// **Strict allow-list:** unknown top-level fields reject with 400
/// ``unsupported_field``. Type-invalid values for accepted fields reject
/// with 400 ``invalid_request``.
///
/// **Known-rejected fields**:
/// - ``functions`` / ``function_call`` — deprecated by OpenAI; use ``tools`` instead.
/// - ``modalities``, ``audio``, ``metadata``, ``store``, ``service_tier``,
///   ``prediction``, ``reasoning_effort``, ``verbosity`` — out of scope.
///
/// **Streaming:** ``stream: true`` is supported (SSE ``chat.completion.chunk``).
/// ``n > 1`` streaming fans candidates out as per-``choice_index``-tagged delta
/// chunks with a per-choice closure carrying ``finish_reason`` (and a per-choice
/// ``delta.role:"assistant"`` once per choice) before the single global ``[DONE]``.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatCompletionMessage>,
    /// Preferred output-token cap. Falls back to ``max_tokens`` when
    /// absent. When BOTH are omitted the gateway applies a default
    /// (1024, override via ``SIE_GATEWAY_DEFAULT_MAX_TOKENS``) rather
    /// than rejecting — matching OpenAI, where this field is optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    /// Legacy compatibility — ``max_completion_tokens`` wins when both
    /// are present. Optional; see ``max_completion_tokens`` for the
    /// behaviour when neither is supplied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Sampling temperature. Finite number ``>= 0``; non-finite values reject.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Nucleus sampling. Finite number in ``(0, 1]``.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Either a string or an array of strings, mirroring OpenAI.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<Value>,
    /// OpenAI ``frequency_penalty`` in ``[-2.0, 2.0]``; out-of-range or
    /// non-numeric values yield 400 ``invalid_request``.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// OpenAI ``presence_penalty`` in ``[-2.0, 2.0]``; same validation as
    /// ``frequency_penalty``.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    /// Non-OpenAI ``top_k`` (Together / Fireworks / vLLM extension):
    /// integer ``>= 1``. Absent → top-k disabled (model default).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Non-OpenAI ``repetition_penalty``: float in ``(0.0, 2.0]``
    /// (``1.0`` = no penalty). Absent → sampler default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,
    /// SGLang ``sampling_params.min_new_tokens``: integer ``>= 0`` capping
    /// how many tokens the model must emit before any stop condition can
    /// fire. Use to work around models that occasionally emit the stop
    /// token as the very first decoded token (e.g. Qwen3.6 thinking-off
    /// under greedy decode). Absent → sampler default (no minimum).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_tokens: Option<u32>,
    /// Per-request overrides for kwargs passed to the tokenizer's
    /// ``apply_chat_template`` call. Object whose entries are merged on
    /// top of the model YAML's ``chat_template_kwargs`` (the YAML wins
    /// for keys present in both — adjust the YAML when a request needs
    /// to defeat a baked-in default). Typical use: pass
    /// ``{"enable_thinking": false}`` to the Qwen3 family to suppress
    /// ``<think>`` reasoning on a per-request basis. Absent → only the
    /// model YAML's defaults apply.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template_kwargs: Option<Value>,
    /// Number of candidate completions, ``[1, 128]``. ``n>1`` returns a
    /// multi-entry ``choices`` array. Streaming with ``n>1`` is supported:
    /// per-``choice_index`` delta chunks plus per-choice closure chunks
    /// (each carrying a ``finish_reason``) ride before the single ``[DONE]``.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    /// OpenAI ``best_of``: integer in ``[1, 128]``. Generate this many
    /// candidates and return the top ``n`` by cumulative logprob.
    /// Cross-field rule: ``best_of >= n``. Rejected with 400
    /// ``unsupported_field`` when ``stream: true`` (mirrors OpenAI).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best_of: Option<u32>,
    /// SIE extension: non-empty served-name of a LoRA adapter declared in the
    /// model profile's ``lora_paths``. Absent → the base model. The gateway
    /// resolves the name against the model's loaded adapters; unknown name →
    /// 400 with ``param:"lora_adapter"``. Forwarded to the worker as a
    /// top-level ``lora_path`` generation kwarg (SGLang then selects the
    /// adapter by served name).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora_adapter: Option<String>,
    /// OpenAI ``user`` — accepted-and-dropped (debug-logged only, never
    /// persisted, never forwarded to the worker).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// Accepted and silently ignored (never logged, never forwarded).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,
    /// Prompt-cache hint; plumbed onto the work envelope and
    /// ignored by the worker on the chat-completions surface.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    /// Routing affinity hint; same plumbing as
    /// ``prompt_cache_key``.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing_key: Option<String>,
    /// Optional signed 64-bit per-request sampling seed. Reproducibility is
    /// best effort, not guaranteed, and backend/deployment-specific.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    /// Return per-token logprobs. Boolean. When ``true``, the chosen
    /// token's logprob (and optionally a top-N list) rides on each
    /// ``choices[].logprobs`` entry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    /// Number of alternative top tokens to return alongside each chosen
    /// token's logprob. Integer in ``[0, 20]``. Requires ``logprobs: true``
    /// when > 0 (the OpenAI rule).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,
    /// OpenAI ``logit_bias`` — ``{token_id_string: bias_float}``. Keys must
    /// parse as integer token ids; values must be finite numbers in
    /// ``[-100.0, 100.0]``. Map size capped at 1024 keys.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<std::collections::BTreeMap<String, f64>>,
    /// OpenAI ``response_format`` — translated into a grammar spec on the
    /// worker. Accepted shapes: ``{type:"text"}``, ``{type:"json_object"}``,
    /// ``{type:"json_schema", json_schema:{...}}``. Cannot be combined with
    /// a forcing ``tool_choice`` (``"required"`` or a named function); two
    /// competing grammars on one request reject with 400 ``invalid_request``.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>,
    /// SSE streaming. When ``true``, the response is a stream of
    /// ``chat.completion.chunk`` events terminated by ``data: [DONE]``.
    /// For ``n > 1``, per-``choice_index`` delta chunks include a per-choice
    /// ``delta.role:"assistant"`` once per choice; per-choice closure chunks
    /// carry the ``finish_reason`` for that choice before the global ``[DONE]``.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// OpenAI ``stream_options``. Accepted sub-key: ``include_usage``
    /// (boolean — when true, the gateway emits a terminal ``usage`` frame
    /// before ``[DONE]``). Any other sub-key rejects with 400
    /// ``unsupported_field``. Legal with ``stream:false`` (the options are
    /// simply ignored).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<Value>,
    /// OpenAI tool-calling. Array of tool specs; each tool must match
    /// ``{type:"function", function:{name, parameters, description?}}``.
    /// Combined with ``n > 1``, per-candidate ``tool_calls`` surface on
    /// ``choices[i].message.tool_calls`` (non-streaming). On the streaming
    /// path, tool-call deltas ride on each ``choices[].delta`` chunk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Value>,
    /// OpenAI ``tool_choice``. One of: ``"auto"``, ``"none"``, ``"required"``,
    /// or ``{type:"function", function:{name}}`` (named function). Requires
    /// ``tools`` to be set. ``"required"`` and named-function choices cannot
    /// be combined with ``response_format`` (two competing grammars).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    /// OpenAI ``parallel_tool_calls`` — boolean controlling whether the
    /// model may emit multiple tool calls per turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ChatCompletionChoiceMessage {
    pub role: String,
    pub content: String,
    /// Per-candidate tool-call list. Present when the model
    /// emitted one or more tool calls for this choice; absent otherwise.
    /// Each entry has the OpenAI shape ``{id, type:"function",
    /// function:{name, arguments}}`` where ``arguments`` is a JSON-encoded string.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ChatCompletionChoice {
    pub index: u32,
    pub message: ChatCompletionChoiceMessage,
    /// One of ``"stop" | "length" | "tool_calls"``. The chat surface collapses
    /// unknown SIE-native finish reasons to ``stop`` so strict OpenAI clients
    /// still parse the response. ``_close_choice`` coerces a length-truncated
    /// candidate that also produced a tool call to ``"tool_calls"`` per the
    /// OpenAI convention.
    pub finish_reason: String,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ChatCompletionUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ChatCompletionResponse {
    /// Always ``"chatcmpl-<request_id>"``.
    pub id: String,
    /// Always ``"chat.completion"`` on the non-streaming endpoint.
    pub object: String,
    /// Epoch seconds.
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: ChatCompletionUsage,
}

/// OpenAI-compatible ``POST /v1/completions`` request (legacy raw-prompt surface).
///
/// **Strict allow-list**: unknown fields reject with 400
/// ``unsupported_field``. Type-invalid values reject with 400 ``invalid_request``.
///
/// **Known-rejected fields**:
/// - ``echo`` — rejected with 400 ``unsupported_field``.
/// - ``suffix`` — rejected with 400 ``unsupported_field``.
/// - ``logprobs`` — rejected with 400 ``unsupported_field`` (the legacy
///   ``{tokens, token_logprobs}`` response shape is a follow-up; chat
///   ``logprobs`` is available on ``/v1/chat/completions``).
/// - ``best_of`` — rejected with 400 ``unsupported_field`` (use chat).
/// - ``n > 1`` — rejected with 400 ``unsupported_field`` (chat is the
///   multi-candidate surface). ``n == 1`` (or absent) is a no-op.
/// - Batched array ``prompt`` — rejected with 400 ``unsupported_field``;
///   send one prompt string.
///
/// **Streaming:** ``stream: true`` is supported (SSE ``text_completion``).
///
/// **Response body wire change:** the always-null ``logprobs`` field has
/// been dropped from the response body; SDKs that destructure
/// ``choices[].logprobs`` should treat absence as the new normal.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct CompletionsRequest {
    pub model: String,
    /// Single prompt string. Batched array prompts reject with 400
    /// ``unsupported_field``.
    pub prompt: String,
    /// Positive integer; defaults to 16 (OpenAI's documented default for
    /// completions) when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Sampling temperature. Finite number ``>= 0``.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Nucleus sampling. Finite number in ``(0, 1]``.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Either a string or an array of strings, mirroring OpenAI.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<Value>,
    /// In ``[-2.0, 2.0]``; out-of-range or non-numeric values yield 400.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// In ``[-2.0, 2.0]``; out-of-range or non-numeric values yield 400.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    /// Optional signed 64-bit per-request sampling seed. Reproducibility is
    /// best effort, not guaranteed, and backend/deployment-specific.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    /// SSE streaming. ``true`` is supported; the response is a stream of
    /// ``text_completion`` events terminated by ``data: [DONE]``.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Single-candidate only: ``1`` (or absent) is accepted; ``n > 1``
    /// rejects with 400 ``unsupported_field`` (use chat for multi-candidate).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
}

/// OpenAI-compatible ``POST /v1/responses`` request (Responses MVP).
///
/// **Strict allow-list**: unknown fields reject with 400
/// ``unsupported_field``. Type-invalid values reject with 400 ``invalid_request``.
///
/// **Known-rejected fields**:
/// - ``tools`` / ``tool_choice`` — rejected with 400 ``unsupported_field``.
/// - ``previous_response_id`` — rejected with 400 ``unsupported_field``
///   (Responses MVP is stateless single-turn).
/// - ``reasoning`` — rejected with 400 ``unsupported_field``.
/// - ``background`` — rejected with 400 ``unsupported_field``.
/// - ``metadata`` — rejected with 400 ``unsupported_field``.
/// - ``instructions`` — rejected with 400 ``unsupported_field``.
/// - ``stream: true`` — rejected with 400 ``unsupported_field`` (SSE on
///   Responses is deferred; use ``stream: false`` or omit).
/// - Multimodal ``image_url`` / ``input_image`` content parts on the array
///   form — rejected with 400 ``unsupported_field``: the Responses surface is
///   text-only. (Vision input is supported on ``/v1/chat/completions``.)
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ResponsesRequest {
    pub model: String,
    /// Either a string prompt OR an array of ``{role, content}`` messages
    /// (array-input support). Array form: ``role`` is one of
    /// ``"system" | "user" | "assistant" | "developer"`` (``"tool"`` is
    /// nominally accepted but Responses tools are rejected so it is
    /// effectively unreachable). ``content`` is a string or an array of
    /// text-only content parts; image parts reject.
    pub input: Value,
    /// Defaults to 16 (mirroring completions) when absent. Positive integer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    /// Sampling temperature. Finite number ``>= 0``.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Nucleus sampling. Finite number in ``(0, 1]``.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Optional signed 64-bit per-request sampling seed. Reproducibility is
    /// best effort, not guaranteed, and backend/deployment-specific.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    /// Accepted only as ``false`` (or absent). ``true`` rejects with 400
    /// ``unsupported_field`` (Responses SSE is deferred).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

/// OpenAI-shaped error body used by ``/v1/generate/{model}`` and
/// ``/v1/chat/completions``.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct OpenAIErrorBody {
    pub message: String,
    /// One of the stable types in ``http_error::openai_type``.
    #[serde(rename = "type")]
    pub err_type: String,
    /// Offending field name (e.g. ``"messages"``, ``"max_completion_tokens"``).
    /// ``null`` when the error is not field-specific.
    pub param: Option<String>,
    /// SIE-native discriminator, see ``http_error::openai_code``.
    pub code: String,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct OpenAIErrorEnvelope {
    pub error: OpenAIErrorBody,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct GatewayErrorResponse {
    #[serde(default)]
    pub message: Option<String>,
    #[serde(default)]
    pub error: Option<Value>,
    #[serde(default)]
    pub details: Option<Vec<InferenceErrorDetail>>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub gpu: Option<String>,
    #[serde(default)]
    pub configured_gpu_types: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum InferenceInternalServerErrorResponse {
    AllItemsFailed(AllItemsFailedResponse),
    Standard(StandardApiError),
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum InferenceServiceUnavailableResponse {
    Gateway(GatewayErrorResponse),
    Standard(StandardApiError),
    GpuNotConfigured(GpuNotConfiguredError),
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct BundleRoutingConflictDetail {
    pub code: String,
    pub message: String,
    pub compatible_bundles: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct BundleConflictResponse {
    pub detail: BundleRoutingConflictDetail,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ResolveModelNotFoundDetail {
    pub code: String,
    pub message: String,
    pub model: String,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ResolveModelNotFoundResponse {
    pub detail: ResolveModelNotFoundDetail,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ResolveBundleConflictDetail {
    pub code: String,
    pub message: String,
    pub model: String,
    pub bundle: String,
    pub compatible_bundles: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ResolveBundleConflictResponse {
    pub detail: ResolveBundleConflictDetail,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct AllItemsFailedResponse {
    pub error: String,
    pub details: Vec<InferenceErrorDetail>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct MessageResponse {
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct HealthResponse {
    pub status: String,
    #[serde(rename = "type")]
    #[schema(rename = "type")]
    pub service_type: String,
    pub configured_gpu_types: Vec<String>,
    pub live_gpu_types: Vec<String>,
    pub cluster: ClusterSummary,
    pub workers: Vec<crate::types::worker::WorkerInfo>,
    pub models: Vec<crate::types::worker::ModelInfo>,
    #[serde(default)]
    pub pending_generation: crate::queue::publisher::PendingGenerationSnapshot,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ClusterSummary {
    pub worker_count: i32,
    pub gpu_count: i32,
    pub models_loaded: i32,
    pub total_qps: f64,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ProfileInfoWire {
    #[serde(default)]
    pub is_default: bool,
}

/// Wire shape aligned with ``sie_server`` ``ModelInfo`` for ``GET /v1/models``.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ModelInfoWire {
    pub name: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub dims: std::collections::HashMap<String, i64>,
    pub loaded: bool,
    pub state: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_error: Option<serde_json::Value>,
    pub max_sequence_length: Option<u64>,
    /// Immutable model weights revision, when the catalog pins one.
    pub revision: Option<String>,
    pub profiles: std::collections::HashMap<String, ProfileInfoWire>,
    /// Advertised model capabilities. Consumers use this to discover
    /// which features the model supports before composing a request.
    ///
    /// Validation is profile-scoped per ADR-0001 / M10 — clients
    /// selecting a specific profile must check
    /// ``capabilities.profile_lora_adapters[profile_name]``, not the
    /// ``capabilities.lora_adapters`` union summary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<ModelCapabilitiesWire>,
    #[serde(default)]
    pub pending_generation: crate::queue::publisher::PendingGenerationSnapshot,
}

/// Capability summary surfaced on each entry of ``GET /v1/models``.
///
/// Mirrors the JSON shape constructed in
/// ``types/model.rs::to_model_info_value``. All fields are optional —
/// their presence depends on what the model config declares.
///
/// These flags indicate that a model *supports* a task — they are NOT a
/// precision-independent quality SLA. A flag is true at the model level even
/// when quality is profile/precision-dependent (e.g. ``sql`` quality regresses
/// under FP8; route SQL-critical traffic to a BF16 bundle via the ``sql``
/// alias). Treat them as "can do this", not "guaranteed to score X".
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ModelCapabilitiesWire {
    /// Union of LoRA served-names across profiles. Back-compat summary
    /// for consumers that don't care about profile scope; validation
    /// MUST go through ``profile_lora_adapters``.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora_adapters: Option<Vec<String>>,
    /// Per-profile LoRA breakdown — keyed by profile name. Added by
    /// M10 so consumers needing precise routing scope don't have to
    /// reverse-engineer it from the union. The validation gate uses
    /// this map; ``lora_adapters`` is for display only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profile_lora_adapters: Option<std::collections::HashMap<String, Vec<String>>>,
    /// Grammar kinds the model's active backend supports
    /// (``json_schema`` | ``regex`` | ``ebnf``). EBNF presence depends
    /// on the backend: SGLang's Outlines backend does not implement
    /// EBNF, so a profile with ``grammar_backend: outlines`` advertises
    /// only ``["json_schema", "regex"]``; xgrammar/llguidance profiles
    /// advertise all three.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grammar: Option<Vec<String>>,
    /// Whether the model supports tool / function calling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<bool>,
    /// Whether the model is validated for code generation — backs the
    /// ``model="code"`` alias. Informational only; never request-gated.
    pub code: bool,
    /// Whether the model targets the SQL path — backs the ``model="sql"``
    /// alias. Informational only; never request-gated. Precision-sensitive:
    /// SQL quality can regress sharply under FP8, so this model-level flag is a
    /// support signal, not a per-profile quality guarantee.
    pub sql: bool,
    /// Whether the model is a content-moderation / policy-check guard —
    /// backs the ``model="guard"`` alias. Informational only; never
    /// request-gated.
    pub guard: bool,
}

/// One OpenAI-shaped model object in the `data` array of
/// `GET /v1/models`. Present for OpenAI-ecosystem compatibility; native
/// SIE consumers read the richer [`ModelInfoWire`] entries under
/// `models` instead.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct OpenAiModelObject {
    /// Model id — the same string `/v1/chat/completions` accepts as `model`.
    pub id: String,
    /// Always `"model"`.
    pub object: String,
    /// Unix epoch seconds. SIE has no per-model creation time, so this
    /// is a fixed sentinel; OpenAI clients use it only for display.
    pub created: i64,
    /// Always `"sie"`.
    pub owned_by: String,
}

/// Response for `GET /v1/models`.
///
/// Hybrid shape: `object` + `data` is the OpenAI list format (consumed
/// by vanilla OpenAI clients and Open WebUI for model discovery);
/// `models` is the richer native shape consumed by the SIE Python/TS
/// SDKs. Both describe the same set of models.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ModelsResponse {
    /// Always `"list"` (OpenAI list envelope).
    pub object: String,
    /// OpenAI-shaped model objects for ecosystem compatibility.
    pub data: Vec<OpenAiModelObject>,
    /// Native SIE model info (capabilities, dims, profiles).
    pub models: Vec<ModelInfoWire>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ModelNotFoundResponse {
    pub detail: ModelNotFoundDetail,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ModelNotFoundDetail {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct PoolListResponse {
    pub pools: Vec<crate::types::pool::Pool>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ConfigModelsResponse {
    pub models: Vec<ConfigModelSummary>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ConfigModelSummary {
    pub model_id: String,
    pub profiles: Vec<String>,
    pub source: String,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ConfigModelDocument {
    pub sie_id: String,
    pub source: String,
    pub bundles: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ModelConfigStatusResponse {
    pub model_id: String,
    pub config_epoch: u64,
    pub all_bundles_acked: bool,
    pub no_bundles: bool,
    pub bundles: Vec<ModelAckBundleStatus>,
    #[serde(default)]
    pub pending_generation: crate::queue::publisher::PendingGenerationSnapshot,
    pub source: String,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ModelAckBundleStatus {
    pub bundle_id: String,
    pub expected_bundle_config_hash: String,
    pub total_eligible_workers: usize,
    pub acked_workers: Vec<String>,
    pub pending_workers: Vec<String>,
    pub acked: bool,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct BundleConfigsResponse {
    pub bundles: Vec<BundleConfigSummary>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct BundleConfigSummary {
    pub bundle_id: String,
    pub engine: String,
    pub priority: i32,
    pub adapter_count: usize,
    pub source: String,
    pub connected_workers: usize,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct BundleConfigDocument {
    pub name: String,
    pub engine: String,
    pub priority: i32,
    pub source: String,
    pub adapters: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ResolveConfigResponse {
    pub model: String,
    pub resolved_bundle: String,
    pub compatible_bundles: Vec<String>,
    pub profiles: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ImageInput {
    pub data: Vec<u8>,
    #[serde(default)]
    pub format: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct DocumentInput {
    pub data: Vec<u8>,
    #[serde(default)]
    pub format: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ItemInput {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub images: Option<Vec<ImageInput>>,
    #[serde(default)]
    pub document: Option<DocumentInput>,
    #[serde(default)]
    pub metadata: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct EncodeParams {
    #[serde(default)]
    pub output_types: Option<Vec<OutputType>>,
    #[serde(default)]
    pub instruction: Option<String>,
    #[serde(default)]
    pub output_dtype: Option<OutputDtype>,
    #[serde(default)]
    pub options: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum OutputType {
    Dense,
    Sparse,
    Multivector,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum OutputDtype {
    Float32,
    Float16,
    Int8,
    Uint8,
    Binary,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct EncodeRequest {
    #[schema(min_items = 1)]
    pub items: Vec<ItemInput>,
    #[serde(default)]
    pub params: Option<EncodeParams>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ExtractParams {
    #[serde(default)]
    pub labels: Option<Vec<String>>,
    #[serde(default)]
    pub output_schema: Option<Value>,
    #[serde(default)]
    pub instruction: Option<String>,
    #[serde(default)]
    pub options: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ExtractRequest {
    #[schema(min_items = 1)]
    pub items: Vec<ItemInput>,
    #[serde(default)]
    pub params: Option<ExtractParams>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ScoreRequest {
    pub query: ItemInput,
    #[schema(min_items = 1, max_items = 1000)]
    pub items: Vec<ItemInput>,
    #[serde(default)]
    pub instruction: Option<String>,
    #[serde(default)]
    pub options: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct RerankRequest {
    /// SIE model id. Blank or whitespace-only values are rejected.
    pub model: String,
    /// Text query. Blank or whitespace-only values are rejected.
    pub query: String,
    /// One to 1,000 nonblank text candidates. Execution is internally batched
    /// within the selected model profile's max_batch_tokens limit.
    #[schema(min_items = 1, max_items = 1000)]
    pub documents: Vec<String>,
    #[serde(default)]
    #[schema(minimum = 1)]
    pub top_n: Option<usize>,
    #[serde(default)]
    pub return_documents: Option<bool>,
    #[serde(default)]
    /// SIE extension. Only profile and max_seq_length are accepted.
    pub options: Option<RerankOptions>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct RerankV2Request {
    /// SIE model id. Blank or whitespace-only values are rejected.
    pub model: String,
    /// Text query. Blank or whitespace-only values are rejected.
    pub query: String,
    /// One to 1,000 nonblank text candidates. Execution is internally batched
    /// within the selected model profile's max_batch_tokens limit.
    #[schema(min_items = 1, max_items = 1000)]
    pub documents: Vec<String>,
    #[serde(default)]
    #[schema(minimum = 1)]
    pub top_n: Option<usize>,
    #[serde(default)]
    pub max_tokens_per_doc: Option<Value>,
    #[serde(default)]
    pub priority: Option<Value>,
    #[serde(default)]
    /// SIE extension. Only profile and max_seq_length are accepted.
    pub options: Option<RerankOptions>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct RerankOptions {
    #[serde(default)]
    /// Named SIE model profile. Blank values are rejected.
    pub profile: Option<String>,
    #[serde(default)]
    /// Positive request-time sequence-length cap, clamped to the model's
    /// configured ceiling. Reranker truncation preserves the instruction and
    /// query and truncates candidate document tokens first.
    #[schema(minimum = 1)]
    pub max_seq_length: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct RerankDocument {
    pub text: String,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct RerankResult {
    pub index: usize,
    pub relevance_score: f64,
    #[serde(default)]
    pub document: Option<RerankDocument>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ScoreUsage {
    pub input_tokens: u64,
    #[serde(default)]
    pub images: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct RerankResponse {
    pub model: String,
    pub results: Vec<RerankResult>,
    pub usage: ScoreUsage,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct RerankError {
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct DenseVector {
    pub dims: usize,
    pub dtype: OutputDtype,
    pub values: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct SparseVector {
    #[serde(default)]
    pub dims: Option<usize>,
    pub dtype: OutputDtype,
    pub indices: Vec<usize>,
    pub values: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct MultiVector {
    pub token_dims: usize,
    pub num_tokens: usize,
    pub dtype: OutputDtype,
    pub values: Vec<Vec<f32>>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct EncodeResult {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub dense: Option<DenseVector>,
    #[serde(default)]
    pub sparse: Option<SparseVector>,
    #[serde(default)]
    pub multivector: Option<MultiVector>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct TimingInfo {
    pub total_ms: f64,
    pub queue_ms: f64,
    pub tokenization_ms: f64,
    pub inference_ms: f64,
    #[serde(default)]
    pub postprocessing_ms: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct EncodeResponse {
    pub model: String,
    pub items: Vec<EncodeResult>,
    #[serde(default)]
    pub timing: Option<TimingInfo>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct Entity {
    pub text: String,
    pub label: String,
    pub score: f64,
    #[serde(default)]
    pub start: Option<usize>,
    #[serde(default)]
    pub end: Option<usize>,
    #[serde(default)]
    pub bbox: Option<Vec<f64>>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct Relation {
    pub head: String,
    pub tail: String,
    pub relation: String,
    pub score: f64,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ExtractItemError {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ExtractResult {
    pub id: String,
    #[serde(default)]
    pub entities: Vec<Entity>,
    #[serde(default)]
    pub relations: Vec<Relation>,
    #[serde(default)]
    pub classifications: Vec<Value>,
    #[serde(default)]
    pub data: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<ExtractItemError>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ExtractResponse {
    pub model: String,
    pub items: Vec<ExtractResult>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ScoreEntry {
    pub item_id: String,
    pub score: f64,
    pub rank: usize,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ScoreResponse {
    pub model: String,
    #[serde(default)]
    pub query_id: Option<String>,
    pub scores: Vec<ScoreEntry>,
    #[serde(default)]
    pub usage: Option<ScoreUsage>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct InferenceErrorDetail {
    pub item_index: u32,
    pub error: Option<String>,
    #[serde(default)]
    pub code: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openapi_document_has_gateway_metadata() {
        let doc = ApiDoc::openapi();
        assert_eq!(doc.info.title, "SIE Gateway");
    }

    #[test]
    fn openapi_documents_rerank_compatibility_contract() {
        let spec: serde_json::Value = serde_json::from_str(&OPENAPI_JSON).unwrap();
        for (path, schema) in [
            ("/v1/rerank", "RerankRequest"),
            ("/v2/rerank", "RerankV2Request"),
        ] {
            let operation = &spec["paths"][path]["post"];
            assert_eq!(
                operation["requestBody"]["content"]["application/json"]["schema"]["$ref"],
                format!("#/components/schemas/{schema}")
            );
            assert_eq!(
                operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"],
                "#/components/schemas/RerankResponse"
            );
            assert_eq!(
                spec["components"]["schemas"][schema]["properties"]["documents"]["maxItems"],
                1000
            );
            assert_eq!(
                spec["components"]["schemas"][schema]["properties"]["documents"]["items"]
                    ["pattern"],
                r".*\S.*"
            );
            assert_eq!(
                spec["components"]["schemas"][schema]["additionalProperties"],
                false
            );
            assert_eq!(
                spec["components"]["schemas"][schema]["properties"]["options"]["oneOf"][1]["$ref"],
                "#/components/schemas/RerankOptions"
            );
            assert!(operation["description"]
                .as_str()
                .unwrap()
                .contains("text-only subset"));
        }
        assert_eq!(
            spec["components"]["schemas"]["RerankOptions"]["additionalProperties"],
            false
        );
        assert_eq!(
            spec["components"]["schemas"]["RerankOptions"]["properties"]["profile"]["pattern"],
            r".*\S.*"
        );
        assert_eq!(
            spec["components"]["schemas"]["RerankV2Request"]["properties"]["priority"]["type"],
            "null"
        );
        assert!(spec["components"]["schemas"]["ScoreResponse"]["properties"]["usage"].is_object());
    }

    #[tokio::test]
    async fn openapi_handler_sets_gateway_version() {
        let doc: serde_json::Value = serde_json::from_str(&OPENAPI_JSON).unwrap();
        assert_eq!(doc["info"]["version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(doc["info"]["license"]["name"], "Apache-2.0");
    }

    #[test]
    fn openapi_json_documents_auth_and_exempt_routes() {
        let spec: serde_json::Value = serde_json::from_str(&OPENAPI_JSON).unwrap();
        assert_eq!(spec["security"], json!([{ "bearerAuth": [] }]));
        assert_eq!(spec["paths"]["/healthz"]["get"]["security"], json!([]));
        assert_eq!(spec["paths"]["/readyz"]["get"]["security"], json!([]));
        assert!(spec["paths"]["/readyz"]["get"]["description"]
            .as_str()
            .unwrap()
            .contains("never returns 503"));
        assert_eq!(spec["paths"]["/openapi.json"]["get"]["security"], json!([]));
        // The rendered reference at /docs must be reachable without a token
        // when auth is on, so client codegen + discovery still work.
        assert_eq!(spec["paths"]["/docs"]["get"]["security"], json!([]));
        assert_eq!(
            spec["components"]["securitySchemes"]["bearerAuth"]["scheme"],
            "bearer"
        );
    }

    #[tokio::test]
    async fn docs_ui_serves_self_contained_redoc_html() {
        let resp = docs_ui().await.into_response();
        assert_eq!(resp.status(), axum::http::StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get(axum::http::header::CONTENT_TYPE)
                .unwrap(),
            "text/html; charset=utf-8"
        );
        let body = axum::body::to_bytes(resp.into_body(), 64 * 1024)
            .await
            .unwrap();
        let html = String::from_utf8(body.to_vec()).unwrap();
        // Renders the live spec via Redoc, loads the vendored (non-CDN) bundle.
        assert!(html.contains("<redoc spec-url=\"openapi.json\">"));
        assert!(html.contains("docs/redoc.standalone.js"));
        assert!(
            !html.contains("cdn.") && !html.contains("http"),
            "no outbound CDN/URL may appear in the served docs page"
        );
    }

    #[tokio::test]
    async fn redoc_asset_serves_vendored_bundle_as_javascript() {
        let resp = redoc_asset().await.into_response();
        assert_eq!(resp.status(), axum::http::StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get(axum::http::header::CONTENT_TYPE)
                .unwrap(),
            "application/javascript; charset=utf-8"
        );
        // The bundle is non-trivially sized (the real Redoc standalone build).
        let body = axum::body::to_bytes(resp.into_body(), 4 * 1024 * 1024)
            .await
            .unwrap();
        assert!(
            body.len() > 100_000,
            "vendored Redoc bundle looks truncated"
        );
    }

    #[test]
    fn openapi_json_documents_its_response_content() {
        let spec: serde_json::Value = serde_json::from_str(&OPENAPI_JSON).unwrap();
        assert_eq!(
            spec["paths"]["/openapi.json"]["get"]["responses"]["200"]["content"]
                ["application/json"]["schema"]["type"],
            "object"
        );
    }

    #[test]
    fn openapi_json_documents_create_pool_runtime_validation() {
        let spec: serde_json::Value = serde_json::from_str(&OPENAPI_JSON).unwrap();
        let create_pool = &spec["components"]["schemas"]["CreatePoolRequest"];
        assert_eq!(create_pool["required"], json!(["name"]));
        assert_eq!(create_pool["properties"]["name"]["minLength"], json!(1));
        assert_eq!(create_pool["properties"]["name"]["maxLength"], json!(128));
        assert_eq!(
            create_pool["properties"]["name"]["pattern"],
            json!("^(?!_[dD][eE][fF][aA][uU][lL][tT]$)[A-Za-z0-9_-]+$")
        );
        assert_eq!(
            create_pool["properties"]["queue_pool"]["minLength"],
            json!(1)
        );
        assert_eq!(
            create_pool["properties"]["queue_pool"]["maxLength"],
            json!(128)
        );
        assert_eq!(
            create_pool["properties"]["queue_pool"]["pattern"],
            json!("^(?!_[dD][eE][fF][aA][uU][lL][tT]$)[A-Za-z0-9_-]+$")
        );
        assert!(create_pool["properties"]["name"]["description"]
            .as_str()
            .unwrap()
            .contains("lowercase"));
        assert_eq!(
            create_pool["anyOf"],
            json!([
                {
                    "required": ["gpus"],
                    "properties": {
                        "gpus": {
                            "minProperties": 1
                        }
                    }
                },
                {
                    "required": ["gpu_caps"],
                    "properties": {
                        "gpu_caps": {
                            "minProperties": 1
                        }
                    }
                }
            ])
        );

        let pool_spec = &spec["components"]["schemas"]["PoolSpec"];
        assert_eq!(pool_spec["properties"]["queue_pool"]["minLength"], json!(1));
        assert_eq!(
            pool_spec["properties"]["queue_pool"]["maxLength"],
            json!(128)
        );
        assert_eq!(
            pool_spec["properties"]["queue_pool"]["pattern"],
            json!("^(?!_[dD][eE][fF][aA][uU][lL][tT]$)[A-Za-z0-9_-]+$")
        );
    }

    #[test]
    fn openapi_json_documents_audio_timestamps() {
        let spec: serde_json::Value = serde_json::from_str(&OPENAPI_JSON).unwrap();
        let request = &spec["components"]["schemas"]["OpenAITranscriptionRequest"]["properties"]
            ["timestamp_granularities[]"];
        assert_eq!(request["maxItems"], json!(2));
        assert_eq!(
            request["items"]["$ref"],
            "#/components/schemas/OpenAITranscriptionTimestampGranularity"
        );
        assert_eq!(
            spec["components"]["schemas"]["OpenAITranscriptionTimestampGranularity"]["enum"],
            json!(["word", "segment"])
        );

        let response = &spec["components"]["schemas"]["OpenAITranscriptionResponse"]["properties"];
        assert_eq!(
            response["segments"]["items"]["$ref"],
            "#/components/schemas/OpenAITranscriptionSegment"
        );
        assert_eq!(
            response["words"]["items"]["$ref"],
            "#/components/schemas/OpenAITranscriptionWord"
        );
        assert_eq!(
            spec["components"]["schemas"]["OpenAITranscriptionWord"]["required"],
            json!(["word", "start", "end"])
        );
        assert_eq!(
            spec["components"]["schemas"]["OpenAITranscriptionSegment"]["required"],
            json!(["id", "start", "end", "text"])
        );
    }

    #[test]
    fn openapi_json_documents_generate_contract() {
        let spec: serde_json::Value = serde_json::from_str(&OPENAPI_JSON).unwrap();
        let generate = &spec["paths"]["/v1/generate/{model}"]["post"];

        assert_eq!(
            generate["requestBody"]["content"]["application/json"]["schema"]["$ref"],
            "#/components/schemas/GenerateRequest"
        );
        assert_eq!(
            generate["responses"]["200"]["content"]["application/json"]["schema"]["$ref"],
            "#/components/schemas/GenerateResponse"
        );
        assert_eq!(
            generate["responses"]["200"]["content"]["text/event-stream"]["schema"]["type"],
            "string"
        );
        assert_eq!(
            generate["responses"]["200"]["content"]["text/event-stream"]["x-sie-event-schema"]
                ["$ref"],
            "#/components/schemas/GenerateChunk"
        );
        assert!(generate["description"]
            .as_str()
            .unwrap()
            .contains("stream: true"));
        assert!(generate["responses"]["200"]["description"]
            .as_str()
            .unwrap()
            .contains("data: [DONE]"));

        let request = &spec["components"]["schemas"]["GenerateRequest"];
        assert_eq!(request["required"], json!(["prompt", "max_new_tokens"]));
        assert_eq!(
            request["properties"]["images"]["items"]["$ref"],
            "#/components/schemas/NativeGenerateImage"
        );
        assert_eq!(request["properties"]["images"]["maxItems"], json!(16));
        assert_eq!(
            spec["components"]["schemas"]["NativeGenerateImage"]["required"],
            json!(["data"])
        );
        assert_eq!(
            spec["components"]["schemas"]["NativeGenerateImage"]["properties"]["data"]["maxLength"],
            json!(22_369_624)
        );
        let seed = &request["properties"]["seed"];
        assert_eq!(seed["type"], json!(["integer", "null"]));
        assert_eq!(seed["format"], "int64");
        assert_eq!(seed["minimum"], i64::MIN);
        assert_eq!(seed["maximum"], i64::MAX);
        assert!(seed["description"]
            .as_str()
            .unwrap()
            .contains("Reproducibility is best effort, not guaranteed"));
        assert_eq!(
            request["properties"]["stream"]["type"],
            json!(["boolean", "null"])
        );
        assert!(request["properties"]["stream"]["description"]
            .as_str()
            .unwrap()
            .contains("text/event-stream"));
        assert_eq!(
            request["properties"]["logprobs"]["type"],
            json!(["boolean", "null"])
        );
        assert_eq!(request["properties"]["top_logprobs"]["maximum"], 20);
        assert_eq!(request["properties"]["lora_adapter"]["minLength"], 1);
        for unsupported in ["n", "best_of", "stream_options"] {
            assert!(request["properties"].get(unsupported).is_none());
        }

        let chunk = &spec["components"]["schemas"]["GenerateChunk"];
        assert_eq!(
            chunk["required"],
            json!(["request_id", "seq", "text_delta", "done"])
        );
        assert_eq!(
            chunk["properties"]["usage"]["oneOf"][1]["$ref"],
            "#/components/schemas/GenerateUsage"
        );
        assert_eq!(
            chunk["properties"]["error"]["oneOf"][1]["$ref"],
            "#/components/schemas/GenerateChunkError"
        );
        assert_eq!(
            chunk["properties"]["logprobs"]["type"],
            json!(["array", "null"])
        );
    }

    #[test]
    fn openapi_json_documents_chat_privacy_and_constraints() {
        let spec: serde_json::Value = serde_json::from_str(&OPENAPI_JSON).unwrap();
        let properties = &spec["components"]["schemas"]["ChatCompletionRequest"]["properties"];

        // n in [1, 128]; n>1 is supported (non-streaming multi-candidate).
        assert_eq!(properties["n"]["minimum"], 1);
        assert_eq!(properties["n"]["maximum"], 128);
        assert_eq!(properties["n"]["type"], "integer");
        assert_eq!(
            properties["stop"]["oneOf"],
            json!([
                {"type": "string"},
                {"type": "array", "items": {"type": "string"}}
            ])
        );
        assert_eq!(properties["user"]["x-sensitive"], true);
        let user_description = properties["user"]["description"].as_str().unwrap();
        assert!(user_description.contains("Sensitive PII"));
        assert!(!user_description.contains("Logged at debug level"));
    }

    #[test]
    fn openapi_document_includes_gateway_paths() {
        let doc = ApiDoc::openapi();
        let paths = doc.paths.paths;
        for path in [
            "/",
            "/openapi.json",
            "/healthz",
            "/v1/models",
            "/v1/pools",
            "/v1/configs/models",
            "/v1/configs/models/{id}/status",
            "/v1/audio/transcriptions",
            "/v1/rerank",
            "/v2/rerank",
            "/v1/embeddings",
            "/v1/encode/{model}",
            "/v1/score/{model}",
            "/v1/extract/{model}",
        ] {
            assert!(paths.contains_key(path), "{path} missing from OpenAPI doc");
        }
    }

    #[test]
    fn openapi_document_covers_all_gateway_route_methods() {
        let spec = serde_json::to_value(openapi_document()).unwrap();
        let paths = spec["paths"].as_object().unwrap();
        let expected = [
            ("/", &["get"][..]),
            ("/healthz", &["get"][..]),
            ("/readyz", &["get"][..]),
            ("/health", &["get"][..]),
            ("/openapi.json", &["get"][..]),
            ("/docs", &["get"][..]),
            ("/v1/models", &["get"][..]),
            ("/v1/models/{model}", &["get"][..]),
            ("/v1/pools", &["get", "post"][..]),
            ("/v1/pools/{name}", &["delete", "get"][..]),
            ("/v1/pools/{name}/renew", &["post"][..]),
            ("/v1/configs/models", &["get"][..]),
            ("/v1/configs/models/{id}", &["get"][..]),
            ("/v1/configs/models/{id}/status", &["get"][..]),
            ("/v1/configs/bundles", &["get"][..]),
            ("/v1/configs/bundles/{id}", &["get"][..]),
            ("/v1/configs/resolve", &["post"][..]),
            ("/v1/audio/transcriptions", &["post"][..]),
            ("/v1/embeddings", &["post"][..]),
            ("/v1/chat/completions", &["post"][..]),
            ("/v1/completions", &["post"][..]),
            ("/v1/responses", &["post"][..]),
            ("/v1/moderations", &["post"][..]),
            ("/v1/rerank", &["post"][..]),
            ("/v2/rerank", &["post"][..]),
            ("/ws/cluster-status", &["get"][..]),
            ("/v1/encode/{model}", &["post"][..]),
            ("/v1/score/{model}", &["post"][..]),
            ("/v1/extract/{model}", &["post"][..]),
            ("/v1/generate/{model}", &["post"][..]),
        ];

        let actual_paths: std::collections::BTreeSet<_> =
            paths.keys().map(String::as_str).collect();
        let expected_paths: std::collections::BTreeSet<_> =
            expected.iter().map(|(path, _)| *path).collect();
        assert_eq!(actual_paths, expected_paths);

        for (path, methods) in expected {
            let path_item = paths[path].as_object().unwrap();
            let actual_methods: std::collections::BTreeSet<_> =
                path_item.keys().map(String::as_str).collect();
            let expected_methods: std::collections::BTreeSet<_> = methods.iter().copied().collect();
            assert_eq!(actual_methods, expected_methods, "{path} methods differ");
        }
    }

    #[test]
    fn openapi_document_does_not_advertise_engine_header_on_inference_paths() {
        let spec = serde_json::to_value(openapi_document()).unwrap();
        for path in [
            "/v1/encode/{model}",
            "/v1/score/{model}",
            "/v1/extract/{model}",
            "/v1/audio/transcriptions",
            "/v1/embeddings",
        ] {
            let parameters = spec["paths"][path]["post"]["parameters"]
                .as_array()
                .unwrap_or_else(|| panic!("{path} must document parameters"));
            assert!(
                !parameters.iter().any(|p| p["name"]
                    .as_str()
                    .is_some_and(|name| name.eq_ignore_ascii_case("x-sie-engine"))),
                "{path} must not advertise X-SIE-Engine; normal routing uses model profile variants"
            );
        }
    }

    #[test]
    fn openapi_document_covers_inference_error_statuses() {
        let spec = serde_json::to_value(openapi_document()).unwrap();
        for path in [
            "/v1/encode/{model}",
            "/v1/score/{model}",
            "/v1/extract/{model}",
        ] {
            let responses = spec["paths"][path]["post"]["responses"]
                .as_object()
                .unwrap();
            for status in [
                "200", "400", "404", "409", "413", "500", "502", "503", "504",
            ] {
                assert!(responses.contains_key(status), "{path} missing {status}");
            }
            assert!(
                !responses.contains_key("202"),
                "{path} must not document 202 provisioning"
            );
        }

        let generate = &spec["paths"]["/v1/generate/{model}"]["post"]["responses"]
            .as_object()
            .unwrap();
        for status in ["200", "400", "404", "413", "500", "502", "503", "504"] {
            assert!(
                generate.contains_key(status),
                "/v1/generate/{{model}} missing {status}"
            );
        }
        assert!(
            !generate.contains_key("202"),
            "/v1/generate/{{model}} must not document 202 provisioning"
        );

        let emb = &spec["paths"]["/v1/embeddings"]["post"]["responses"]
            .as_object()
            .unwrap();
        for status in [
            "200", "400", "401", "404", "409", "413", "500", "502", "503", "504",
        ] {
            assert!(emb.contains_key(status), "/v1/embeddings missing {status}");
        }
        assert!(
            !emb.contains_key("202"),
            "/v1/embeddings must not document 202 as a success-like provisioning response"
        );
    }

    #[test]
    fn openapi_document_documents_inference_error_unions() {
        let spec = serde_json::to_value(openapi_document()).unwrap();
        let internal_error =
            &spec["components"]["schemas"]["InferenceInternalServerErrorResponse"]["oneOf"];
        assert_eq!(
            internal_error,
            &json!([
                {"$ref": "#/components/schemas/AllItemsFailedResponse"},
                {"$ref": "#/components/schemas/StandardApiError"},
            ])
        );

        let service_unavailable =
            &spec["components"]["schemas"]["InferenceServiceUnavailableResponse"]["oneOf"];
        assert_eq!(
            service_unavailable,
            &json!([
                {"$ref": "#/components/schemas/GatewayErrorResponse"},
                {"$ref": "#/components/schemas/StandardApiError"},
                {"$ref": "#/components/schemas/GpuNotConfiguredError"},
            ])
        );

        // SIE-native passthrough endpoints keep the legacy `detail`-shaped
        // error unions. `/v1/embeddings` is OpenAI-shaped and asserted
        // separately below.
        for path in [
            "/v1/encode/{model}",
            "/v1/score/{model}",
            "/v1/extract/{model}",
        ] {
            assert_eq!(
                spec["paths"][path]["post"]["responses"]["500"]["content"]["application/json"]
                    ["schema"]["$ref"],
                "#/components/schemas/InferenceInternalServerErrorResponse"
            );
            assert_eq!(
                spec["paths"][path]["post"]["responses"]["503"]["content"]["application/json"]
                    ["schema"]["$ref"],
                "#/components/schemas/InferenceServiceUnavailableResponse"
            );
        }

        // `/v1/embeddings` is OpenAI-compatible: every handler-generated error
        // path documents the OpenAI `{error:{…}}` envelope so an `openai`
        // client's error handling works unchanged. (401 is emitted upstream by
        // the auth middleware and stays on the SIE-native shape.)
        // (In the raw document each error is a direct `$ref`; the export step
        // later merges the auth-misconfigured `StandardApiError` into the 500
        // as a `oneOf` — see `openapi_export_*` tests.)
        let emb = &spec["paths"]["/v1/embeddings"]["post"]["responses"];
        for status in ["400", "404", "409", "413", "500", "502", "503", "504"] {
            assert_eq!(
                emb[status]["content"]["application/json"]["schema"]["$ref"],
                "#/components/schemas/OpenAIErrorEnvelope",
                "/v1/embeddings {status} must be the OpenAI envelope"
            );
        }
    }

    #[test]
    fn openapi_export_adds_msgpack_content_and_unauthorized_responses() {
        let spec: serde_json::Value = serde_json::from_str(&OPENAPI_JSON).unwrap();
        for path in [
            "/v1/encode/{model}",
            "/v1/score/{model}",
            "/v1/extract/{model}",
        ] {
            let post = &spec["paths"][path]["post"];
            let content = &post["responses"]["200"]["content"];
            assert!(
                content.get("application/json").is_some(),
                "{path} missing application/json 200"
            );
            assert!(
                content.get("application/x-msgpack").is_some(),
                "{path} missing application/x-msgpack 200"
            );
            assert!(
                content.get("application/msgpack").is_some(),
                "{path} missing application/msgpack 200"
            );
            assert!(
                content.get("application/vnd.msgpack").is_some(),
                "{path} missing application/vnd.msgpack 200"
            );
            let req_content = &post["requestBody"]["content"];
            assert!(req_content.get("application/x-msgpack").is_some());
            assert!(req_content.get("application/msgpack").is_some());
            assert!(req_content.get("application/vnd.msgpack").is_some());

            let responses = post["responses"].as_object().unwrap();
            assert!(responses.contains_key("401"), "{path} missing injected 401");
            assert!(
                responses.contains_key("500"),
                "{path} missing auth-misconfigured 500"
            );
            assert!(
                responses["200"]["headers"]
                    .get("X-SIE-Request-Id")
                    .is_some(),
                "{path} missing documented request id header"
            );
            assert!(
                responses["200"]["headers"].get("X-Queue-Time").is_some(),
                "{path} missing documented queue timing header"
            );
            assert!(
                responses["200"]["headers"]
                    .get("X-SIE-Model-Revision")
                    .is_some(),
                "{path} missing documented model revision header"
            );
            assert!(
                responses["200"]["headers"]
                    .get("X-SIE-Execution-Identity-SHA256")
                    .is_some(),
                "{path} missing documented worker execution identity header"
            );
        }

        let pool_post = &spec["paths"]["/v1/pools"]["post"]["responses"];
        assert!(pool_post.as_object().unwrap().contains_key("401"));
        assert!(pool_post.as_object().unwrap().contains_key("403"));
        assert!(pool_post.as_object().unwrap().contains_key("500"));

        let config_resolve = &spec["paths"]["/v1/configs/resolve"]["post"]["responses"];
        assert!(config_resolve.as_object().unwrap().contains_key("401"));
        assert!(config_resolve.as_object().unwrap().contains_key("403"));
        assert!(config_resolve.as_object().unwrap().contains_key("500"));

        for path in [
            "/v1/encode/{model}",
            "/v1/score/{model}",
            "/v1/extract/{model}",
            "/v1/generate/{model}",
            "/v1/audio/transcriptions",
            "/v1/embeddings",
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/responses",
        ] {
            let retryable_503_headers = &spec["paths"][path]["post"]["responses"]["503"]["headers"];
            assert!(
                retryable_503_headers.get("Retry-After").is_some(),
                "{path} 503 missing Retry-After"
            );
            assert!(
                retryable_503_headers.get("X-SIE-Error-Code").is_some(),
                "{path} 503 missing X-SIE-Error-Code"
            );
            assert!(
                retryable_503_headers.get("X-SIE-Version").is_some(),
                "{path} 503 missing X-SIE-Version"
            );
            assert!(
                retryable_503_headers.get("X-SIE-Server-Version").is_some(),
                "{path} 503 missing X-SIE-Server-Version"
            );
        }

        let embeddings_200_headers =
            &spec["paths"]["/v1/embeddings"]["post"]["responses"]["200"]["headers"];
        for name in [
            "X-SIE-Request-Id",
            "X-SIE-Worker",
            "X-Queue-Publish-Time",
            "X-Queue-Wait-Time",
            "X-Queue-Time",
            "X-Inference-Time",
            "X-Tokenization-Time",
            "X-Postprocessing-Time",
            "X-Payload-Fetch-Time",
        ] {
            assert!(
                embeddings_200_headers.get(name).is_some(),
                "/v1/embeddings 200 missing documented {name} header"
            );
        }
        for path in ["/v1/chat/completions", "/v1/completions", "/v1/responses"] {
            let headers = &spec["paths"][path]["post"]["responses"]["200"]["headers"];
            for name in [
                "X-SIE-Request-Id",
                "X-SIE-Model-Revision",
                "X-SIE-Execution-Identity-SHA256",
            ] {
                assert!(
                    headers.get(name).is_some(),
                    "{path} 200 missing documented {name} header"
                );
            }
            assert_eq!(
                headers["X-SIE-Execution-Identity-SHA256"]["schema"]["pattern"],
                "^[0-9a-f]{64}$"
            );
        }
    }

    #[test]
    fn openapi_export_documents_server_parity_cleanup() {
        let spec: serde_json::Value = serde_json::from_str(&OPENAPI_JSON).unwrap();

        for schema in ["EncodeResponse", "ExtractResponse", "ScoreResponse"] {
            assert!(
                spec["components"]["schemas"][schema]["properties"]
                    .get("errors")
                    .is_none(),
                "{schema} must not document gateway-only partial error envelope"
            );
        }

        let extract_result = &spec["components"]["schemas"]["ExtractResult"];
        assert_eq!(
            extract_result["properties"]["error"]["oneOf"][1]["$ref"],
            "#/components/schemas/ExtractItemError"
        );
        assert!(!extract_result["required"]
            .as_array()
            .unwrap()
            .iter()
            .any(|field| field == "error"));
        assert!(spec["paths"]["/v1/extract/{model}"]["post"]["description"]
            .as_str()
            .unwrap()
            .contains("aligned per-item error"));

        let model_param = &spec["paths"]["/v1/encode/{model}"]["post"]["parameters"][0];
        assert!(model_param.get("allowReserved").is_none());
        assert!(model_param["description"]
            .as_str()
            .unwrap()
            .contains("percent-encode slashes"));
        assert_eq!(
            spec["paths"]["/v1/encode/{model}"]["post"]["x-sie-axum-catch-all"],
            "/v1/encode/{*model}"
        );
        let generate_model_param = &spec["paths"]["/v1/generate/{model}"]["post"]["parameters"][0];
        assert!(generate_model_param["description"]
            .as_str()
            .unwrap()
            .contains("percent-encode slashes"));
        assert_eq!(
            spec["paths"]["/v1/generate/{model}"]["post"]["x-sie-axum-catch-all"],
            "/v1/generate/{*model}"
        );

        let model_load_failed = &spec["components"]["schemas"]["GatewayModelLoadFailedResponse"];
        assert!(model_load_failed["properties"].get("error").is_some());
        assert!(model_load_failed["properties"].get("detail").is_none());

        let embeddings_request = &spec["components"]["schemas"]["OpenAIEmbeddingRequest"];
        assert_eq!(
            embeddings_request["properties"]["input"]["$ref"],
            "#/components/schemas/OpenAIEmbeddingInput"
        );
        assert!(embeddings_request["properties"]["encoding_format"]["oneOf"]
            .as_array()
            .unwrap()
            .iter()
            .any(|schema| schema["$ref"] == "#/components/schemas/OpenAIEmbeddingEncodingFormat"));
        assert_eq!(
            embeddings_request["properties"]["dimensions"]["type"],
            json!(["integer", "null"])
        );
        assert_eq!(
            embeddings_request["properties"]["user"]["type"],
            json!(["string", "null"])
        );
        assert_eq!(
            spec["components"]["schemas"]["OpenAIEmbeddingEncodingFormat"]["enum"],
            json!(["float", "base64"])
        );

        for schema in ["EncodeRequest", "ExtractRequest"] {
            assert_eq!(
                spec["components"]["schemas"][schema]["properties"]["items"]["maxItems"],
                json!(crate::queue::publisher::MAX_QUEUE_REQUEST_ITEMS),
                "{schema} must document the runtime queue-item limit",
            );
        }
        assert_eq!(
            spec["components"]["schemas"]["ScoreRequest"]["properties"]["items"]["maxItems"],
            json!(crate::handlers::proxy::MAX_SCORE_ITEMS),
            "ScoreRequest must document the stricter runtime score-item limit",
        );
        let embedding_input_variants = spec["components"]["schemas"]["OpenAIEmbeddingInput"]
            ["oneOf"]
            .as_array()
            .expect("embedding input must be a union");
        let embedding_array = embedding_input_variants
            .iter()
            .find(|variant| variant["type"] == "array")
            .expect("embedding input must document the string-array form");
        assert_eq!(
            embedding_array["maxItems"],
            json!(crate::queue::publisher::MAX_QUEUE_REQUEST_ITEMS),
        );

        for path in ["/v1/encode/{model}", "/v1/score/{model}"] {
            assert!(spec["paths"][path]["post"]["description"]
                .as_str()
                .unwrap()
                .contains("Mixed-success batches return 200"));
        }
        assert!(spec["paths"]["/v1/embeddings"]["post"]["description"]
            .as_str()
            .unwrap()
            .contains("partial or truncated internal encode success"));

        let headers = &spec["paths"]["/v1/embeddings"]["post"]["responses"]["502"]["headers"];
        assert!(headers.get("X-SIE-Error-Code").is_some());
        assert!(headers.get("X-SIE-Error-Version").is_some());
    }
}
