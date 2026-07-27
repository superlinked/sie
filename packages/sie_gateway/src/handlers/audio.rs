//! OpenAI audio transcription compatibility over native `extract`.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::extract::multipart::Field;
use axum::extract::{FromRequest, Multipart, Request, State};
use axum::http::{header, HeaderMap, HeaderValue, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;
use serde_json::{json, Map, Value};

use crate::http_error::{
    code as err_code, embeddings_error, json_openai_error, openai_code as oai_code,
    openai_type as oai_type,
};
use crate::server::AppState;

use super::proxy::{
    is_openai_compat_forwarded_header, is_openai_compat_inner_request_header,
    is_valid_compat_model_id, proxy_request, translate_inner_compat_error,
};

const MAX_AUDIO_FILE_BYTES: usize = 24 * 1024 * 1024;
pub(crate) const MAX_MULTIPART_BYTES: usize = MAX_AUDIO_FILE_BYTES + 1024 * 1024;
const MAX_TEXT_FIELD_BYTES: usize = 8 * 1024;
const MAX_FORM_FIELDS: usize = 16;
const MAX_NATIVE_RESPONSE_BYTES: usize = 34 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResponseFormat {
    Json,
    Text,
    Srt,
    VerboseJson,
    Vtt,
}

#[derive(Debug)]
struct TranscriptionForm {
    audio: Vec<u8>,
    audio_format: String,
    model: String,
    language: Option<String>,
    prompt: Option<String>,
    response_format: ResponseFormat,
    temperature: Option<f64>,
    timestamp_granularities: Vec<String>,
}

#[derive(Serialize)]
struct NativeExtractAudio<'a> {
    #[serde(with = "serde_bytes")]
    data: &'a [u8],
    format: &'a str,
}

#[derive(Serialize)]
struct NativeExtractItem<'a> {
    audio: NativeExtractAudio<'a>,
}

#[derive(Serialize)]
struct NativeExtractParams<'a> {
    instruction: Option<&'a str>,
    options: &'a Map<String, Value>,
}

#[derive(Serialize)]
struct NativeExtractRequest<'a> {
    items: [NativeExtractItem<'a>; 1],
    params: NativeExtractParams<'a>,
}

#[derive(Debug)]
struct FormError {
    status: StatusCode,
    message: String,
    param: Option<String>,
    code: &'static str,
}

impl FormError {
    fn invalid(param: Option<&str>, message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            param: param.map(str::to_string),
            code: oai_code::INVALID_REQUEST,
        }
    }

    fn unsupported(param: &str, message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            param: Some(param.to_string()),
            code: oai_code::UNSUPPORTED_FIELD,
        }
    }

    fn payload_too_large(param: Option<&str>, message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::PAYLOAD_TOO_LARGE,
            message: message.into(),
            param: param.map(str::to_string),
            code: oai_code::PAYLOAD_TOO_LARGE,
        }
    }

    fn into_response(self) -> Response {
        (
            self.status,
            Json(json_openai_error(
                self.message,
                oai_type::INVALID_REQUEST,
                self.param.as_deref(),
                self.code,
            )),
        )
            .into_response()
    }
}

async fn read_field(mut field: Field<'_>, limit: usize, param: &str) -> Result<Vec<u8>, FormError> {
    let mut bytes = Vec::new();
    loop {
        let chunk = field.chunk().await.map_err(|_| {
            FormError::invalid(Some(param), "request must be valid multipart/form-data")
        })?;
        let Some(chunk) = chunk else {
            break;
        };
        if bytes.len().saturating_add(chunk.len()) > limit {
            return Err(FormError::payload_too_large(
                Some(param),
                format!("field '{param}' exceeds {limit} bytes"),
            ));
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

async fn read_text_field(field: Field<'_>, name: &str) -> Result<String, FormError> {
    let bytes = read_field(field, MAX_TEXT_FIELD_BYTES, name).await?;
    String::from_utf8(bytes)
        .map_err(|_| FormError::invalid(Some(name), format!("field '{name}' must be UTF-8 text")))
}

fn supported_extension(filename: &str) -> Option<String> {
    let extension = Path::new(filename)
        .extension()
        .and_then(|value| value.to_str())?
        .to_ascii_lowercase();
    matches!(
        extension.as_str(),
        "flac" | "mp3" | "mp4" | "mpeg" | "mpga" | "m4a" | "ogg" | "wav" | "webm"
    )
    .then_some(extension)
}

async fn parse_transcription_form(
    mut multipart: Multipart,
) -> Result<TranscriptionForm, FormError> {
    let mut audio: Option<Vec<u8>> = None;
    let mut audio_format: Option<String> = None;
    let mut values: HashMap<String, String> = HashMap::new();
    let mut granularities = Vec::new();
    let mut field_count = 0usize;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|_| FormError::invalid(None, "request must be valid multipart/form-data"))?
    {
        field_count += 1;
        if field_count > MAX_FORM_FIELDS {
            return Err(FormError::invalid(
                None,
                "multipart form contains too many fields",
            ));
        }
        let name = field
            .name()
            .map(str::to_string)
            .ok_or_else(|| FormError::invalid(None, "multipart field is missing a name"))?;
        match name.as_str() {
            "file" => {
                if audio.is_some() {
                    return Err(FormError::invalid(
                        Some("file"),
                        "field 'file' must appear once",
                    ));
                }
                let filename = field.file_name().ok_or_else(|| {
                    FormError::invalid(Some("file"), "field 'file' must be a file upload")
                })?;
                let extension = supported_extension(filename).ok_or_else(|| {
                    FormError::invalid(
                        Some("file"),
                        "unsupported audio file format; expected flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm",
                    )
                })?;
                let bytes = read_field(field, MAX_AUDIO_FILE_BYTES, "file").await?;
                if bytes.is_empty() {
                    return Err(FormError::invalid(
                        Some("file"),
                        "field 'file' must not be empty",
                    ));
                }
                audio = Some(bytes);
                audio_format = Some(extension);
            }
            "timestamp_granularities" | "timestamp_granularities[]" => {
                granularities.push(read_text_field(field, "timestamp_granularities").await?);
            }
            "model" | "language" | "prompt" | "response_format" | "temperature" | "stream" => {
                if values.contains_key(&name) {
                    return Err(FormError::invalid(
                        Some(&name),
                        format!("field '{name}' must appear once"),
                    ));
                }
                values.insert(name.clone(), read_text_field(field, &name).await?);
            }
            _ => {
                return Err(FormError::unsupported(
                    &name,
                    format!("unsupported field '{name}'"),
                ));
            }
        }
    }

    let audio =
        audio.ok_or_else(|| FormError::invalid(Some("file"), "field 'file' is required"))?;
    let audio_format = audio_format.expect("audio format is set with audio bytes");
    let model = values
        .remove("model")
        .unwrap_or_default()
        .trim()
        .to_string();
    if model.is_empty() {
        return Err(FormError::invalid(
            Some("model"),
            "field 'model' is required",
        ));
    }
    if !is_valid_compat_model_id(&model) {
        return Err(FormError::invalid(
            Some("model"),
            "invalid model id for path",
        ));
    }

    let language = values
        .remove("language")
        .map(|value| value.trim().to_string());
    if language.as_deref() == Some("") {
        return Err(FormError::invalid(
            Some("language"),
            "field 'language' must not be empty",
        ));
    }
    let prompt = values.remove("prompt");

    let response_format = match values
        .remove("response_format")
        .unwrap_or_else(|| "json".to_string())
        .as_str()
    {
        "json" => ResponseFormat::Json,
        "text" => ResponseFormat::Text,
        "srt" => ResponseFormat::Srt,
        "verbose_json" => ResponseFormat::VerboseJson,
        "vtt" => ResponseFormat::Vtt,
        "diarized_json" => {
            return Err(FormError::unsupported(
                "response_format",
                "response_format 'diarized_json' is not supported by this model",
            ));
        }
        _ => {
            return Err(FormError::invalid(
                Some("response_format"),
                "response_format must be json, text, srt, verbose_json, or vtt",
            ));
        }
    };

    match values.remove("stream").as_deref().unwrap_or("false") {
        "false" => {}
        "true" => {
            return Err(FormError::unsupported(
                "stream",
                "streaming transcription is not supported by the native extract primitive",
            ));
        }
        _ => {
            return Err(FormError::invalid(
                Some("stream"),
                "field 'stream' must be true or false",
            ));
        }
    }

    let temperature = values
        .remove("temperature")
        .map(|value| {
            value.parse::<f64>().map_err(|_| {
                FormError::invalid(
                    Some("temperature"),
                    "temperature must be a number between 0 and 1",
                )
            })
        })
        .transpose()?;
    if temperature.is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value)) {
        return Err(FormError::invalid(
            Some("temperature"),
            "temperature must be a number between 0 and 1",
        ));
    }

    if granularities
        .iter()
        .any(|value| value != "word" && value != "segment")
    {
        return Err(FormError::invalid(
            Some("timestamp_granularities"),
            "timestamp_granularities must contain only 'word' or 'segment'",
        ));
    }
    let mut unique_granularities = Vec::with_capacity(granularities.len());
    for granularity in granularities {
        if !unique_granularities.contains(&granularity) {
            unique_granularities.push(granularity);
        }
    }
    granularities = unique_granularities;
    if !granularities.is_empty() && response_format != ResponseFormat::VerboseJson {
        return Err(FormError::invalid(
            Some("timestamp_granularities"),
            "timestamp_granularities requires response_format='verbose_json'",
        ));
    }
    if matches!(response_format, ResponseFormat::Srt | ResponseFormat::Vtt) {
        granularities = vec!["segment".to_string()];
    }

    Ok(TranscriptionForm {
        audio,
        audio_format,
        model,
        language,
        prompt,
        response_format,
        temperature,
        timestamp_granularities: granularities,
    })
}

fn native_extract_msgpack(form: &TranscriptionForm) -> Result<Vec<u8>, rmp_serde::encode::Error> {
    let mut options = Map::new();
    options.insert(
        "timestamp_granularities".to_string(),
        json!(form.timestamp_granularities),
    );
    if let Some(language) = &form.language {
        options.insert("language".to_string(), json!(language));
    }
    if let Some(temperature) = form.temperature {
        options.insert("temperature".to_string(), json!(temperature));
    }
    rmp_serde::to_vec_named(&NativeExtractRequest {
        items: [NativeExtractItem {
            audio: NativeExtractAudio {
                data: &form.audio,
                format: &form.audio_format,
            },
        }],
        params: NativeExtractParams {
            instruction: form.prompt.as_deref(),
            options: &options,
        },
    })
}

fn duration_ms(data: &Map<String, Value>) -> Result<u64, &'static str> {
    data.get("duration_ms")
        .and_then(Value::as_u64)
        .filter(|duration_ms| *duration_ms > 0)
        .ok_or("extract response is missing a positive integer duration_ms")
}

fn timestamp(value: &Value, separator: char) -> Result<String, &'static str> {
    let seconds = value
        .as_f64()
        .filter(|value| value.is_finite() && *value >= 0.0)
        .ok_or("extract response contains an invalid timestamp")?;
    let total_ms = (seconds * 1000.0).round() as u64;
    let hours = total_ms / 3_600_000;
    let remainder = total_ms % 3_600_000;
    let minutes = remainder / 60_000;
    let remainder = remainder % 60_000;
    let seconds = remainder / 1_000;
    let milliseconds = remainder % 1_000;
    Ok(format!(
        "{hours:02}:{minutes:02}:{seconds:02}{separator}{milliseconds:03}"
    ))
}

fn subtitle(data: &Map<String, Value>, vtt: bool) -> Result<String, &'static str> {
    let segments = data
        .get("segments")
        .and_then(Value::as_array)
        .ok_or("extract response is missing segment timestamps")?;
    let mut blocks = Vec::with_capacity(segments.len());
    for (index, segment) in segments.iter().enumerate() {
        let segment = segment
            .as_object()
            .ok_or("extract response contains an invalid segment")?;
        let separator = if vtt { '.' } else { ',' };
        let start = timestamp(segment.get("start").unwrap_or(&Value::Null), separator)?;
        let end = timestamp(segment.get("end").unwrap_or(&Value::Null), separator)?;
        let text = segment
            .get("text")
            .and_then(Value::as_str)
            .ok_or("extract response contains invalid segment text")?
            .trim();
        let prefix = if vtt {
            String::new()
        } else {
            format!("{}\n", index + 1)
        };
        blocks.push(format!("{prefix}{start} --> {end}\n{text}"));
    }
    let mut content = blocks.join("\n\n");
    content.push('\n');
    if vtt {
        content.insert_str(0, "WEBVTT\n\n");
    }
    Ok(content)
}

fn format_transcription_response(
    data: &Map<String, Value>,
    form: &TranscriptionForm,
) -> Result<Response, &'static str> {
    let text = data
        .get("text")
        .and_then(Value::as_str)
        .ok_or("extract response is missing text")?;
    let duration_ms = duration_ms(data)?;
    let duration = duration_ms as f64 / 1000.0;
    let response = match form.response_format {
        ResponseFormat::Json => Json(json!({
            "text": text,
            "usage": {"type": "duration", "seconds": duration},
        }))
        .into_response(),
        ResponseFormat::Text => (
            [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
            text.to_string(),
        )
            .into_response(),
        ResponseFormat::Srt => (
            [(header::CONTENT_TYPE, "application/x-subrip")],
            subtitle(data, false)?,
        )
            .into_response(),
        ResponseFormat::Vtt => (
            [(header::CONTENT_TYPE, "text/vtt; charset=utf-8")],
            subtitle(data, true)?,
        )
            .into_response(),
        ResponseFormat::VerboseJson => {
            let language = data
                .get("language")
                .and_then(Value::as_str)
                .or(form.language.as_deref())
                .unwrap_or("unknown");
            let mut output = json!({
                "task": "transcribe",
                "language": language,
                "duration": duration,
                "text": text,
                "usage": {"type": "duration", "seconds": duration},
            });
            let object = output
                .as_object_mut()
                .expect("verbose response is an object");
            if form
                .timestamp_granularities
                .iter()
                .any(|value| value == "segment")
            {
                object.insert(
                    "segments".to_string(),
                    data.get("segments").cloned().unwrap_or_else(|| json!([])),
                );
            }
            if form
                .timestamp_granularities
                .iter()
                .any(|value| value == "word")
            {
                object.insert(
                    "words".to_string(),
                    data.get("words").cloned().unwrap_or_else(|| json!([])),
                );
            }
            Json(output).into_response()
        }
    };
    Ok(response)
}

fn copy_inner_headers(inbound: &HeaderMap) -> HeaderMap {
    let mut headers = HeaderMap::new();
    for (name, value) in inbound {
        if is_openai_compat_inner_request_header(name.as_str()) {
            headers.append(name.clone(), value.clone());
        }
    }
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/x-msgpack"),
    );
    headers.insert(header::ACCEPT, HeaderValue::from_static("application/json"));
    headers
}

fn copy_native_response_headers(from: &HeaderMap, to: &mut HeaderMap) {
    for (name, value) in from {
        if is_openai_compat_forwarded_header(name.as_str()) {
            to.insert(name.clone(), value.clone());
        }
    }
}

#[utoipa::path(
    post,
    path = "/v1/audio/transcriptions",
    tag = "inference",
    description = "OpenAI-compatible audio transcription backed by SIE native extract. The multipart upload is bounded at 24 MiB. json, text, srt, verbose_json, and vtt responses are supported; stream=true and diarized_json are rejected explicitly.",
    request_body(content = crate::openapi::OpenAITranscriptionRequest, content_type = "multipart/form-data"),
    params(
        ("X-SIE-MACHINE-PROFILE" = Option<String>, Header, description = "Preferred GPU or machine profile"),
        ("X-SIE-Pool" = Option<String>, Header, description = "Explicit pool routing override"),
        ("X-SIE-SDK-Version" = Option<String>, Header, description = "Client SDK version for skew warnings")
    ),
    responses(
        (status = 200, description = "OpenAI-compatible transcription response", body = crate::openapi::OpenAITranscriptionResponse),
        (status = 400, description = "Invalid or unsupported field", body = crate::openapi::OpenAIErrorEnvelope),
        (status = 401, description = "Missing or invalid bearer token", body = crate::openapi::StandardApiError),
        (status = 404, description = "Model not found", body = crate::openapi::OpenAIErrorEnvelope),
        (status = 413, description = "Multipart body or audio file too large", body = crate::openapi::OpenAIErrorEnvelope),
        (status = 500, description = "Malformed worker response or gateway internal error", body = crate::openapi::OpenAIErrorEnvelope),
        (status = 502, description = "Model load failed", body = crate::openapi::OpenAIErrorEnvelope),
        (status = 503, description = "Provisioning, queue, loading, or capacity unavailable", body = crate::openapi::OpenAIErrorEnvelope),
        (status = 504, description = "Result channel closed", body = crate::openapi::OpenAIErrorEnvelope)
    )
)]
pub async fn proxy_openai_transcription(
    State(state): State<Arc<AppState>>,
    req: Request,
) -> Response {
    if req
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<usize>().ok())
        .is_some_and(|value| value > MAX_MULTIPART_BYTES)
    {
        return FormError::payload_too_large(
            None,
            "multipart body exceeds the 25 MiB ingress limit",
        )
        .into_response();
    }

    let inbound_headers = req.headers().clone();
    let version = req.version();
    let extensions = req.extensions().clone();
    let multipart = match Multipart::from_request(req, &state).await {
        Ok(multipart) => multipart,
        Err(rejection) => {
            let status = rejection.status();
            return if status == StatusCode::PAYLOAD_TOO_LARGE {
                FormError::payload_too_large(
                    None,
                    "multipart body exceeds the 25 MiB ingress limit",
                )
                .into_response()
            } else {
                FormError::invalid(None, "request must be valid multipart/form-data")
                    .into_response()
            };
        }
    };
    let form = match parse_transcription_form(multipart).await {
        Ok(form) => form,
        Err(error) => return error.into_response(),
    };

    let body = match native_extract_msgpack(&form) {
        Ok(body) => body,
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(embeddings_error(
                    err_code::INTERNAL_ERROR,
                    None,
                    "failed to serialize native extract request",
                )),
            )
                .into_response();
        }
    };
    let uri = match format!("/v1/extract/{}", form.model).parse::<axum::http::Uri>() {
        Ok(uri) => uri,
        Err(_) => {
            return FormError::invalid(Some("model"), "invalid model id for path").into_response()
        }
    };
    let mut builder = Request::builder()
        .method(Method::POST)
        .uri(uri)
        .version(version);
    for (name, value) in copy_inner_headers(&inbound_headers) {
        if let Some(name) = name {
            builder = builder.header(name, value);
        }
    }
    let mut inner_request = match builder.body(Body::from(body)) {
        Ok(request) => request,
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(embeddings_error(
                    err_code::INTERNAL_ERROR,
                    None,
                    "failed to build native extract request",
                )),
            )
                .into_response();
        }
    };
    *inner_request.extensions_mut() = extensions;

    let native_response = proxy_request(State(state), inner_request, "extract").await;
    if native_response.status().is_client_error() || native_response.status().is_server_error() {
        return translate_inner_compat_error(native_response).await;
    }
    if native_response.status() != StatusCode::OK {
        return native_response;
    }

    let native_headers = native_response.headers().clone();
    let bytes = match to_bytes(native_response.into_body(), MAX_NATIVE_RESPONSE_BYTES).await {
        Ok(bytes) => bytes,
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(embeddings_error(
                    err_code::INTERNAL_ERROR,
                    None,
                    "failed to read native extract response",
                )),
            )
                .into_response();
        }
    };
    let parsed: Value = match serde_json::from_slice(&bytes) {
        Ok(value) => value,
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(embeddings_error(
                    err_code::INTERNAL_ERROR,
                    None,
                    "native extract response is not valid JSON",
                )),
            )
                .into_response();
        }
    };
    let data = parsed
        .get("items")
        .and_then(Value::as_array)
        .filter(|items| items.len() == 1)
        .and_then(|items| items[0].get("data"))
        .and_then(Value::as_object);
    let Some(data) = data else {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(embeddings_error(
                err_code::INTERNAL_ERROR,
                None,
                "native extract response is missing one transcription result",
            )),
        )
            .into_response();
    };
    let mut response = match format_transcription_response(data, &form) {
        Ok(response) => response,
        Err(message) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(embeddings_error(err_code::INTERNAL_ERROR, None, message)),
            )
                .into_response();
        }
    };
    copy_native_response_headers(&native_headers, response.headers_mut());
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    fn multipart_body(fields: &[(&str, &str)], filename: &str, audio: &[u8]) -> (String, Vec<u8>) {
        let boundary = "sie-audio-test";
        let mut body = Vec::new();
        for (name, value) in fields {
            body.extend_from_slice(
                format!(
                    "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
                )
                .as_bytes(),
            );
        }
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(audio);
        body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
        (format!("multipart/form-data; boundary={boundary}"), body)
    }

    async fn parse_body(
        fields: &[(&str, &str)],
        filename: &str,
    ) -> Result<TranscriptionForm, FormError> {
        let (content_type, body) = multipart_body(fields, filename, b"RIFFtest");
        let request = Request::builder()
            .header(header::CONTENT_TYPE, content_type)
            .body(Body::from(body))
            .unwrap();
        let multipart = Multipart::from_request(request, &()).await.unwrap();
        parse_transcription_form(multipart).await
    }

    #[test]
    fn internal_extract_delegation_is_binary_msgpack_with_bounded_headers() {
        let form = TranscriptionForm {
            audio: vec![0x5a; MAX_AUDIO_FILE_BYTES],
            audio_format: "wav".to_string(),
            model: "openai/whisper-large-v3-turbo".to_string(),
            language: Some("en".to_string()),
            prompt: Some("SIE".to_string()),
            response_format: ResponseFormat::Json,
            temperature: Some(0.25),
            timestamp_granularities: vec!["word".to_string()],
        };

        let body = native_extract_msgpack(&form).unwrap();
        assert!(body.len() <= form.audio.len() + 1024);
        let decoded: rmpv::Value = rmp_serde::from_slice(&body).unwrap();
        let rmpv::Value::Map(root) = &decoded else {
            panic!("request must be a map");
        };
        let items = root
            .iter()
            .find_map(|(key, value)| {
                matches!(key, rmpv::Value::String(key) if key.as_str() == Some("items"))
                    .then_some(value)
            })
            .unwrap();
        let rmpv::Value::Array(items) = items else {
            panic!("items must be an array");
        };
        let rmpv::Value::Map(item) = items.first().unwrap() else {
            panic!("item must be a map");
        };
        let audio = item
            .iter()
            .find_map(|(key, value)| {
                matches!(key, rmpv::Value::String(key) if key.as_str() == Some("audio"))
                    .then_some(value)
            })
            .unwrap();
        let rmpv::Value::Map(audio) = audio else {
            panic!("audio must be a map");
        };
        let data = audio
            .iter()
            .find_map(|(key, value)| {
                matches!(key, rmpv::Value::String(key) if key.as_str() == Some("data"))
                    .then_some(value)
            })
            .unwrap();
        let rmpv::Value::Binary(data) = data else {
            panic!("audio.data must be MessagePack binary");
        };
        assert_eq!(data.as_slice(), form.audio.as_slice());

        let mut inbound = HeaderMap::new();
        inbound.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer test"),
        );
        inbound.insert(
            "traceparent",
            HeaderValue::from_static("00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"),
        );
        inbound.insert(header::COOKIE, HeaderValue::from_static("secret=drop"));
        inbound.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("multipart/form-data; boundary=test"),
        );
        let inner = copy_inner_headers(&inbound);
        assert_eq!(
            inner.get(header::CONTENT_TYPE).unwrap(),
            "application/x-msgpack"
        );
        assert_eq!(inner.get(header::ACCEPT).unwrap(), "application/json");
        assert_eq!(inner.get(header::AUTHORIZATION).unwrap(), "Bearer test");
        assert!(inner.contains_key("traceparent"));
        assert!(!inner.contains_key(header::COOKIE));

        let mut native = HeaderMap::new();
        native.insert("x-sie-request-id", HeaderValue::from_static("req-1"));
        native.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );
        let mut compat = HeaderMap::new();
        copy_native_response_headers(&native, &mut compat);
        assert_eq!(compat.get("x-sie-request-id").unwrap(), "req-1");
        assert!(!compat.contains_key(header::CONTENT_TYPE));
    }

    #[tokio::test]
    async fn multipart_parser_accepts_supported_fields() {
        let form = parse_body(
            &[
                ("model", "openai/whisper-large-v3-turbo"),
                ("language", "en"),
                ("prompt", "SIE"),
                ("temperature", "0.25"),
                ("response_format", "verbose_json"),
                ("timestamp_granularities[]", "word"),
                ("timestamp_granularities[]", "word"),
                ("timestamp_granularities[]", "segment"),
            ],
            "clip.WAV",
        )
        .await
        .unwrap();

        assert_eq!(form.audio, b"RIFFtest");
        assert_eq!(form.audio_format, "wav");
        assert_eq!(form.language.as_deref(), Some("en"));
        assert_eq!(form.prompt.as_deref(), Some("SIE"));
        assert_eq!(form.temperature, Some(0.25));
        assert_eq!(form.timestamp_granularities, ["word", "segment"]);

        let default_verbose = parse_body(
            &[("model", "whisper"), ("response_format", "verbose_json")],
            "clip.wav",
        )
        .await
        .unwrap();
        assert!(default_verbose.timestamp_granularities.is_empty());

        assert_eq!(
            duration_ms(json!({"duration_ms": 1}).as_object().unwrap()),
            Ok(1)
        );
        for invalid in [json!({"duration_ms": 0}), json!({"duration_ms": 1.5})] {
            assert_eq!(
                duration_ms(invalid.as_object().unwrap()),
                Err("extract response is missing a positive integer duration_ms")
            );
        }
    }

    #[tokio::test]
    async fn multipart_parser_rejects_unknown_and_streaming_fields() {
        let unknown = parse_body(
            &[("model", "whisper"), ("chunking_strategy", "auto")],
            "clip.wav",
        )
        .await
        .unwrap_err();
        assert_eq!(unknown.param.as_deref(), Some("chunking_strategy"));
        assert_eq!(unknown.code, oai_code::UNSUPPORTED_FIELD);

        let streaming = parse_body(&[("model", "whisper"), ("stream", "true")], "clip.wav")
            .await
            .unwrap_err();
        assert_eq!(streaming.param.as_deref(), Some("stream"));
        assert_eq!(streaming.code, oai_code::UNSUPPORTED_FIELD);

        let oversized_prompt = "x".repeat(MAX_TEXT_FIELD_BYTES + 1);
        let oversized = parse_body(
            &[("model", "whisper"), ("prompt", &oversized_prompt)],
            "clip.wav",
        )
        .await
        .unwrap_err();
        assert_eq!(oversized.status, StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(oversized.code, oai_code::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn multipart_parser_rejects_ambiguous_model_paths() {
        for model in [
            "/leading",
            "two..dots",
            "back\\slash",
            "query?x=1",
            "fragment#x",
            "unicode-model-模型",
        ] {
            let error = parse_body(&[("model", model)], "clip.wav")
                .await
                .unwrap_err();
            assert_eq!(error.param.as_deref(), Some("model"), "{model}");
        }
        let error = FormError::payload_too_large(Some("file"), "too large");
        assert_eq!(error.code, oai_code::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn formats_json_and_subtitles_from_native_extract_data() {
        let data = json!({
            "text": "hello world",
            "language": "english",
            "duration_ms": 1234,
            "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "hello world"}],
            "words": [{"word": "hello", "start": 0.0, "end": 0.5}],
        });
        let data = data.as_object().unwrap();
        let mut form = parse_body(&[("model", "whisper")], "clip.wav")
            .await
            .unwrap();

        let json_response = format_transcription_response(data, &form).unwrap();
        let body = to_bytes(json_response.into_body(), 4096).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&body).unwrap(),
            json!({"text": "hello world", "usage": {"type": "duration", "seconds": 1.234}})
        );

        form.response_format = ResponseFormat::VerboseJson;
        form.timestamp_granularities = vec!["word".to_string(), "segment".to_string()];
        let verbose = format_transcription_response(data, &form).unwrap();
        let body = to_bytes(verbose.into_body(), 4096).await.unwrap();
        let verbose: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            verbose["words"],
            json!([{"word": "hello", "start": 0.0, "end": 0.5}])
        );
        assert_eq!(
            verbose["segments"],
            json!([{"id": 0, "start": 0.0, "end": 1.0, "text": "hello world"}])
        );

        form.response_format = ResponseFormat::Srt;
        let srt = format_transcription_response(data, &form).unwrap();
        let body = to_bytes(srt.into_body(), 4096).await.unwrap();
        assert_eq!(body, "1\n00:00:00,000 --> 00:00:01,000\nhello world\n");
    }
}
