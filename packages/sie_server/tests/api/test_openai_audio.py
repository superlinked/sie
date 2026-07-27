"""Contract tests for the OpenAI-compatible transcription adapter."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sie_server.api import openai_audio
from sie_server.config.model import ExtractTask, InputModalities, ModelConfig, ProfileConfig, Tasks
from sie_server.core.inference_output import ExtractOutput
from sie_server.core.registry import ModelRegistry
from sie_server.core.timing import RequestTiming
from sie_server.core.worker import WorkerResult

MODEL = "openai/whisper-large-v3-turbo"
AUDIO = b"RIFF\x00\x00\x00\x00WAVEtest"


def _model_config(*, audio: bool = True, extract: bool = True) -> ModelConfig:
    return ModelConfig(
        sie_id=MODEL,
        hf_id=MODEL,
        inputs=InputModalities(text=False, audio=audio),
        tasks=Tasks(extract=ExtractTask() if extract else None),
        profiles={
            "default": ProfileConfig(
                adapter_path="sie_server.adapters.whisper.adapter:WhisperAdapter",
                max_batch_tokens=720_000,
            )
        },
    )


@pytest.fixture
def registry() -> MagicMock:
    registry = MagicMock(spec=ModelRegistry)
    registry.has_model.return_value = True
    registry.is_loaded.return_value = True
    registry.is_loading.return_value = False
    registry.is_unloading.return_value = False
    registry.is_failed.return_value = False
    registry.get_failure.return_value = None
    registry.get_worker.return_value = None
    registry.get_config.return_value = _model_config()
    registry.device = "cpu"
    registry.engine_config = None
    return registry


@pytest.fixture
def extract_mock(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    output = ExtractOutput(
        entities=[[]],
        data=[
            {
                "text": "hello world",
                "language": "english",
                "duration_ms": 1_234,
                "words": [
                    {"word": " hello", "start": 0.0, "end": 0.5},
                    {"word": " world", "start": 0.5, "end": 1.0},
                ],
                "segments": [
                    {"id": 0, "start": 0.0, "end": 1.0, "text": "hello world"},
                ],
            }
        ],
        batch_size=1,
    )
    mock = AsyncMock(return_value=WorkerResult(output=output, timing=RequestTiming()))
    monkeypatch.setattr(openai_audio, "_extract_via_worker", mock)
    return mock


@pytest.fixture
def client(registry: MagicMock, extract_mock: AsyncMock) -> TestClient:
    del extract_mock
    app = FastAPI()
    app.include_router(openai_audio.router)
    app.state.registry = registry
    return TestClient(app, raise_server_exceptions=False)


def _post(
    client: TestClient,
    *,
    data: dict[str, Any] | None = None,
    filename: str = "clip.wav",
    audio: bytes = AUDIO,
) -> Any:
    return client.post(
        "/v1/audio/transcriptions",
        data={"model": MODEL, **(data or {})},
        files={"file": (filename, audio, "application/octet-stream")},
    )


class TestTranscriptionSuccess:
    def test_json_uses_native_extract_and_exact_duration_usage(
        self,
        client: TestClient,
        extract_mock: AsyncMock,
    ) -> None:
        from openai.types.audio import Transcription

        response = _post(
            client,
            data={
                "language": "en",
                "prompt": "SIE vocabulary",
                "temperature": "0.25",
            },
        )

        assert response.status_code == 200
        assert response.json() == {
            "text": "hello world",
            "usage": {"type": "duration", "seconds": 1.234},
        }
        parsed = Transcription.model_validate(response.json())
        assert parsed.text == "hello world"
        assert parsed.usage is not None
        assert parsed.usage.seconds == 1.234
        assert "x-sie-server-version" in response.headers

        args = extract_mock.await_args
        assert args.args[1] == MODEL
        assert len(args.args[2]) == 1
        assert args.args[2][0].audio is not None
        assert args.args[2][0].audio["data"] == AUDIO
        assert args.args[2][0].audio["format"] == "wav"
        assert args.kwargs["instruction"] == "SIE vocabulary"
        assert args.kwargs["options"] == {
            "language": "en",
            "temperature": 0.25,
            "timestamp_granularities": [],
        }

    @pytest.mark.parametrize(
        ("response_format", "content_type", "expected"),
        [
            ("text", "text/plain", "hello world"),
            ("srt", "application/x-subrip", "1\n00:00:00,000 --> 00:00:01,000\nhello world\n"),
            ("vtt", "text/vtt", "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhello world\n"),
        ],
    )
    def test_text_and_subtitle_formats(
        self,
        client: TestClient,
        response_format: str,
        content_type: str,
        expected: str,
    ) -> None:
        response = _post(client, data={"response_format": response_format})

        assert response.status_code == 200
        assert response.headers["content-type"].startswith(content_type)
        assert response.text == expected

    def test_verbose_json_honors_requested_word_timestamps(self, client: TestClient) -> None:
        from openai.types.audio import TranscriptionVerbose

        response = _post(
            client,
            data={
                "response_format": "verbose_json",
                "timestamp_granularities[]": "word",
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["task"] == "transcribe"
        assert body["language"] == "english"
        assert body["duration"] == 1.234
        assert body["usage"] == {"type": "duration", "seconds": 1.234}
        assert body["words"] == [
            {"word": " hello", "start": 0.0, "end": 0.5},
            {"word": " world", "start": 0.5, "end": 1.0},
        ]
        assert "segments" not in body
        parsed = TranscriptionVerbose.model_validate(body)
        assert parsed.words is not None
        assert parsed.words[0].word == " hello"

    def test_verbose_json_honors_requested_segment_timestamps(
        self,
        client: TestClient,
        extract_mock: AsyncMock,
    ) -> None:
        response = _post(
            client,
            data={
                "response_format": "verbose_json",
                "timestamp_granularities[]": "segment",
            },
        )

        assert response.status_code == 200
        assert response.json()["segments"] == [
            {"id": 0, "start": 0.0, "end": 1.0, "text": "hello world"},
        ]
        assert extract_mock.await_args.kwargs["options"]["timestamp_granularities"] == ["segment"]

    def test_malformed_native_response_is_not_recorded_as_success(
        self,
        client: TestClient,
        extract_mock: AsyncMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        extract_mock.return_value.output.data[0]["duration_ms"] = 0
        telemetry = MagicMock()
        monkeypatch.setattr(openai_audio, "worker_telemetry_enabled", lambda: True)
        monkeypatch.setattr(openai_audio, "worker_telemetry", lambda: telemetry)

        assert _post(client).status_code == 500
        telemetry.item_completed.assert_not_called()

    def test_success_uses_canonical_worker_telemetry_and_checks_sdk_version(
        self,
        client: TestClient,
        extract_mock: AsyncMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        timing = MagicMock()
        timing.total_ms = 100.0
        timing.tokenization_ms = 20.0
        timing.inference_ms = 70.0
        timing.postprocessing_ms = 10.0
        timing.to_headers.return_value = {}
        extract_mock.return_value.timing = timing
        telemetry = MagicMock()
        sdk_version_check = MagicMock()
        monkeypatch.setattr(openai_audio, "worker_telemetry_enabled", lambda: True)
        monkeypatch.setattr(openai_audio, "worker_telemetry", lambda: telemetry)
        monkeypatch.setattr(openai_audio, "check_sdk_version", sdk_version_check)

        assert _post(client).status_code == 200

        sdk_version_check.assert_called_once()
        telemetry.item_completed.assert_called_once_with(
            operation="extract",
            outcome="success",
            model=MODEL,
            profile="default",
            duration_s=0.1,
            item_count=1,
            tokenization_s=0.02,
            inference_s=0.07,
            postprocessing_s=0.01,
        )

    def test_verbose_json_has_no_default_timestamp_granularity(
        self,
        client: TestClient,
        extract_mock: AsyncMock,
    ) -> None:
        response = _post(client, data={"response_format": "verbose_json"})

        assert response.status_code == 200
        assert "words" not in response.json()
        assert "segments" not in response.json()
        assert extract_mock.await_args.kwargs["options"]["timestamp_granularities"] == []

    def test_requires_positive_integer_native_duration(self) -> None:
        assert openai_audio._duration_seconds({"duration_ms": 1}) == 0.001
        for invalid in (0, 1.5):
            with pytest.raises(ValueError, match="positive integer duration_ms"):
                openai_audio._duration_seconds({"duration_ms": invalid})


class TestTranscriptionValidation:
    @pytest.mark.parametrize(
        ("data", "param", "code"),
        [
            ({"stream": "true"}, "stream", "unsupported_field"),
            ({"response_format": "diarized_json"}, "response_format", "unsupported_field"),
            ({"temperature": "nan"}, "temperature", "invalid_request"),
            ({"temperature": "1.1"}, "temperature", "invalid_request"),
            (
                {"response_format": "json", "timestamp_granularities[]": "word"},
                "timestamp_granularities",
                "invalid_request",
            ),
            ({"unknown": "value"}, "unknown", "unsupported_field"),
            ({"model": "/leading"}, "model", "invalid_request"),
            ({"model": "two..dots"}, "model", "invalid_request"),
            ({"model": "back\\slash"}, "model", "invalid_request"),
            ({"model": "query?x=1"}, "model", "invalid_request"),
            ({"model": "fragment#x"}, "model", "invalid_request"),
            ({"model": "unicode-model-模型"}, "model", "invalid_request"),
        ],
    )
    def test_strict_field_validation(
        self,
        client: TestClient,
        data: dict[str, Any],
        param: str,
        code: str,
    ) -> None:
        response = _post(client, data=data)

        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert error["param"] == param
        assert error["code"] == code

    @pytest.mark.parametrize("filename", ["clip.exe", "clip", "clip.WAV.exe"])
    def test_rejects_unknown_file_extensions(self, client: TestClient, filename: str) -> None:
        response = _post(client, filename=filename)

        assert response.status_code == 400
        assert response.json()["error"]["param"] == "file"

    def test_rejects_empty_audio(self, client: TestClient) -> None:
        response = _post(client, audio=b"")

        assert response.status_code == 400
        assert response.json()["error"]["param"] == "file"

    def test_rejects_body_above_declared_limit(self, client: TestClient) -> None:
        response = client.post(
            "/v1/audio/transcriptions",
            headers={
                "content-type": "multipart/form-data; boundary=x",
                "content-length": str(openai_audio._MAX_MULTIPART_BYTES + 1),
            },
            content=b"",
        )

        assert response.status_code == 413
        assert response.json()["error"]["code"] == "payload_too_large"

    def test_rejects_oversized_text_field_as_payload_too_large(self, client: TestClient) -> None:
        response = _post(client, data={"prompt": "x" * (openai_audio._MAX_TEXT_FIELD_BYTES + 1)})

        assert response.status_code == 413
        assert response.json()["error"]["code"] == "payload_too_large"

    def test_file_can_exceed_text_field_limit(
        self,
        client: TestClient,
        extract_mock: AsyncMock,
    ) -> None:
        audio = b"x" * (openai_audio._MAX_TEXT_FIELD_BYTES + 1)

        assert _post(client, audio=audio).status_code == 200
        assert extract_mock.await_args.args[2][0].audio["data"] == audio

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_nonfinite_native_timestamps(self, value: float) -> None:
        with pytest.raises(ValueError, match="invalid timestamp"):
            openai_audio._timestamp(value, separator=".")

    def test_model_not_found_uses_openai_error_envelope(self, client: TestClient, registry: MagicMock) -> None:
        registry.has_model.return_value = False

        response = _post(client)

        assert response.status_code == 404
        assert response.json()["error"] == {
            "message": f"Model '{MODEL}' not found",
            "type": "invalid_request_error",
            "param": None,
            "code": "model_not_found",
        }

    @pytest.mark.parametrize(
        "config",
        [_model_config(audio=False), _model_config(extract=False)],
    )
    def test_rejects_models_without_audio_extract(
        self,
        client: TestClient,
        registry: MagicMock,
        config: ModelConfig,
    ) -> None:
        registry.get_config.return_value = config

        response = _post(client)

        assert response.status_code == 400
        assert response.json()["error"]["param"] == "model"
