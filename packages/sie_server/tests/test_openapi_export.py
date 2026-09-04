import json
import tomllib
from importlib.metadata import version as pkg_version
from pathlib import Path

from sie_server.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_openapi_stdout() -> None:
    """CLI outputs valid OpenAPI JSON to stdout."""
    result = runner.invoke(app, ["openapi"])
    assert result.exit_code == 0, result.output
    spec = json.loads(result.output)
    assert spec["openapi"].startswith("3.")
    assert spec["info"]["title"] == "SIE Server"


def test_openapi_has_expected_paths() -> None:
    """Exported spec contains all core API paths."""
    result = runner.invoke(app, ["openapi"])
    spec = json.loads(result.output)
    paths = set(spec["paths"].keys())
    for expected in [
        "/v1/encode/{model}",
        "/v1/extract/{model}",
        "/v1/generate/{model}",
        "/v1/completions",
        "/v1/responses",
        "/v1/score/{model}",
        "/v1/models",
    ]:
        assert expected in paths, f"Missing path: {expected}"


def test_openapi_has_request_body_schemas() -> None:
    """Custom Pydantic request body schemas are injected."""
    result = runner.invoke(app, ["openapi"])
    spec = json.loads(result.output)
    schemas = spec.get("components", {}).get("schemas", {})
    for name in ["EncodeRequestModel", "ExtractRequestModel", "GenerateRequestModel", "ScoreRequestModel"]:
        assert name in schemas, f"Missing schema: {name}"


def test_openapi_documents_generate_contract() -> None:
    """Worker OpenAPI documents both blocking and streaming native generate."""
    result = runner.invoke(app, ["openapi"])
    assert result.exit_code == 0, result.output
    spec = json.loads(result.output)

    operation = spec["paths"]["/v1/generate/{model}"]["post"]
    request_schema = operation["requestBody"]["content"]["application/json"]["schema"]
    assert request_schema == {"$ref": "#/components/schemas/GenerateRequestModel"}

    schema = spec["components"]["schemas"]["GenerateRequestModel"]
    images = schema["properties"]["images"]
    assert images["anyOf"][0]["items"]["$ref"] == "#/components/schemas/NativeGenerateImageModel"
    assert images["anyOf"][0]["minItems"] == 1
    assert images["anyOf"][0]["maxItems"] == 16
    image_schema = spec["components"]["schemas"]["NativeGenerateImageModel"]
    assert image_schema["properties"]["data"]["maxLength"] == 22_369_624
    grammar = schema["properties"]["grammar"]
    grammar_refs = {variant["$ref"] for variant in grammar["anyOf"] if "$ref" in variant}
    assert grammar_refs == {
        "#/components/schemas/NativeJsonSchemaGrammarModel",
        "#/components/schemas/NativeRegexGrammarModel",
        "#/components/schemas/NativeEbnfGrammarModel",
    }
    assert set(schema["required"]) == {"prompt", "max_new_tokens"}
    assert schema["properties"]["stream"]["anyOf"][0] == {"type": "boolean"}
    seed_schema = schema["properties"]["seed"]
    assert seed_schema["anyOf"][0]["minimum"] == -(1 << 63)
    assert seed_schema["anyOf"][0]["maximum"] == (1 << 63) - 1
    assert seed_schema["format"] == "int64"
    assert schema["properties"]["logit_bias"]["anyOf"][0]["type"] == "object"
    assert schema["properties"]["logprobs"]["anyOf"][0] == {"type": "boolean"}
    assert schema["properties"]["top_logprobs"]["anyOf"][0]["minimum"] == 0
    assert schema["properties"]["top_logprobs"]["anyOf"][0]["maximum"] == 20
    for unsupported in ("lora_adapter", "n", "best_of", "stream_options"):
        assert unsupported not in schema["properties"]

    response_content = operation["responses"]["200"]["content"]
    assert response_content["application/json"]["schema"] == {"$ref": "#/components/schemas/GenerateResponseModel"}
    event_stream = response_content["text/event-stream"]
    assert event_stream["schema"]["type"] == "string"
    assert event_stream["x-sie-event-schema"] == {"$ref": "#/components/schemas/GenerateChunk"}

    chunk_schema = spec["components"]["schemas"]["GenerateChunk"]
    assert set(chunk_schema["required"]) == {"request_id", "seq", "text_delta", "done"}
    assert chunk_schema["properties"]["usage"]["anyOf"][0] == {"$ref": "#/components/schemas/GenerateUsageModel"}
    assert chunk_schema["properties"]["error"]["anyOf"][0] == {"$ref": "#/components/schemas/GenerateChunkErrorModel"}
    assert chunk_schema["properties"]["logprobs"]["anyOf"][0]["type"] == "array"
    chunk_error_schema = spec["components"]["schemas"]["GenerateChunkErrorModel"]
    assert chunk_error_schema["properties"]["param"]["anyOf"][0] == {"type": "string"}
    assert "param" not in chunk_error_schema["required"]
    retry_after = chunk_error_schema["properties"]["retry_after_s"]
    assert retry_after["anyOf"][0]["type"] == "integer"
    assert retry_after["anyOf"][0]["minimum"] == 1
    assert retry_after["anyOf"][0]["maximum"] == 60
    assert "retry_after_s" not in chunk_error_schema["required"]

    responses = operation["responses"]
    assert "INPUT_TOO_LONG" in responses["413"]["description"]
    assert responses["413"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/GenerateInputTooLongErrorResponse"
    }
    assert "MODEL_LOAD_FAILED" in responses["502"]["description"]
    assert "No Retry-After" in responses["502"]["description"]
    assert responses["502"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/GenerateModelLoadFailedErrorResponse"
    }

    input_too_long_detail = spec["components"]["schemas"]["GenerateInputTooLongDetailModel"]
    assert set(input_too_long_detail["required"]) == {"code", "message"}
    assert input_too_long_detail["properties"]["code"]["const"] == "INPUT_TOO_LONG"
    model_load_failed_detail = spec["components"]["schemas"]["GenerateModelLoadFailedDetailModel"]
    assert set(model_load_failed_detail["required"]) == {
        "code",
        "message",
        "error_class",
        "permanent",
        "attempts",
    }
    assert model_load_failed_detail["properties"]["code"]["const"] == "MODEL_LOAD_FAILED"


def test_openapi_documents_streaming_model_capability() -> None:
    result = runner.invoke(app, ["openapi"])
    assert result.exit_code == 0, result.output
    spec = json.loads(result.output)
    streaming = spec["components"]["schemas"]["ModelCapabilities"]["properties"]["streaming"]
    assert streaming["type"] == "boolean"
    assert streaming["default"] is True


def test_openapi_documents_direct_completions_contract() -> None:
    result = runner.invoke(app, ["openapi"])
    assert result.exit_code == 0, result.output
    spec = json.loads(result.output)

    operation = spec["paths"]["/v1/completions"]["post"]
    assert operation["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/OpenAICompletionRequestModel"
    }
    schema = spec["components"]["schemas"]["OpenAICompletionRequestModel"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"model", "prompt"}
    assert schema["properties"]["max_tokens"]["anyOf"][0]["minimum"] == 1
    assert schema["properties"]["stream"]["anyOf"][0] == {"type": "boolean"}
    response_content = operation["responses"]["200"]["content"]
    assert response_content["application/json"]["schema"] == {
        "$ref": "#/components/schemas/OpenAICompletionResponseModel"
    }
    assert response_content["text/event-stream"]["schema"]["type"] == "string"


def test_openapi_documents_direct_responses_contract() -> None:
    result = runner.invoke(app, ["openapi"])
    assert result.exit_code == 0, result.output
    spec = json.loads(result.output)

    operation = spec["paths"]["/v1/responses"]["post"]
    assert operation["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/OpenAIResponsesRequestModel"
    }
    schema = spec["components"]["schemas"]["OpenAIResponsesRequestModel"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"model", "input"}
    assert schema["properties"]["max_output_tokens"]["anyOf"][0]["minimum"] == 1
    assert schema["properties"]["stream"]["anyOf"][0]["const"] is False
    assert operation["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/OpenAIResponsesResponseModel"
    }
    assert "413" in operation["responses"]


def test_openapi_audio_timestamp_contract() -> None:
    """Audio compatibility documents word and segment timestamps."""
    result = runner.invoke(app, ["openapi"])
    assert result.exit_code == 0, result.output
    spec = json.loads(result.output)
    request = spec["paths"]["/v1/audio/transcriptions"]["post"]["requestBody"]
    granularities = request["content"]["multipart/form-data"]["schema"]["properties"]["timestamp_granularities[]"]
    assert granularities["maxItems"] == 2
    assert granularities["items"]["enum"] == ["word", "segment"]


def test_openapi_documents_media_bytes_as_base64_strings() -> None:
    """Every media `data` field advertises the one JSON encoding the API accepts.

    On the JSON path msgspec base64-decodes `data` (matching the msgpack
    path's native binary), so the schema has to say `contentEncoding: base64`.
    Pydantic's default rendering of `bytes` is `format: binary`, which in
    OpenAPI means raw octets -- a generated client that believed it would send
    bytes that never decode.
    """
    result = runner.invoke(app, ["openapi"])
    assert result.exit_code == 0, result.output
    schemas = json.loads(result.output)["components"]["schemas"]
    for name in ["ImageInputModel", "AudioInputModel", "VideoInputModel", "DocumentInputModel"]:
        data = schemas[name]["properties"]["data"]
        assert data["type"] == "string", f"{name}.data must be a string: {data}"
        assert data["contentEncoding"] == "base64", f"{name}.data must declare base64 encoding: {data}"
        assert "format" not in data, f"{name}.data must not claim `format` (binary means raw octets): {data}"


def test_openapi_documents_positive_audio_sample_rate() -> None:
    """`sample_rate` must advertise the positive bound the preprocessor enforces."""
    result = runner.invoke(app, ["openapi"])
    assert result.exit_code == 0, result.output
    schemas = json.loads(result.output)["components"]["schemas"]
    sample_rate = schemas["AudioInputModel"]["properties"]["sample_rate"]
    integer_branch = next(branch for branch in sample_rate["anyOf"] if branch.get("type") == "integer")
    assert integer_branch["exclusiveMinimum"] == 0, f"sample_rate must be positive: {sample_rate}"


def test_openapi_item_accepts_all_media_inputs() -> None:
    """The item schema exposes every media input the worker `Item` accepts."""
    result = runner.invoke(app, ["openapi"])
    assert result.exit_code == 0, result.output
    schemas = json.loads(result.output)["components"]["schemas"]
    properties = schemas["ItemModel"]["properties"]
    for field in ["images", "audio", "video", "document"]:
        assert field in properties, f"ItemModel must document `{field}`"


def test_openapi_output_file(tmp_path: Path) -> None:
    """CLI writes spec to a file when --output is given."""
    out = tmp_path / "spec.json"
    result = runner.invoke(app, ["openapi", "--output", str(out)])
    assert result.exit_code == 0, result.output
    spec = json.loads(out.read_text())
    assert spec["openapi"].startswith("3.")


def test_openapi_version_from_package() -> None:
    """Generated and committed specs match the repo-managed package version."""
    result = runner.invoke(app, ["openapi"])
    assert result.exit_code == 0, result.output
    spec = json.loads(result.output)
    package_dir = Path(__file__).parents[1]
    project = tomllib.loads((package_dir / "pyproject.toml").read_text())
    project_version = project["project"]["version"]
    committed_spec = json.loads((package_dir / "openapi.json").read_text())

    assert pkg_version("sie-server") == project_version
    assert spec["info"]["version"] == project_version
    assert committed_spec["info"]["version"] == project_version
