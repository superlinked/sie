"""The direct extract and transcription routes refuse audio over the 12-minute bound.

Both routes decode audio through ``AudioPreprocessor`` before they submit to the
worker, and the ``sie_audio_prep`` extension refuses anything longer than
``MAX_AUDIO_MS``. The routes, ``_extract_via_worker`` and ``AudioPreprocessor``
are the real ones; only the worker is stubbed, so a call on it stands for audio
reaching the adapter.

The public test environment installs the workspace without the compiled
extension, so each case runs against a stand-in with the extension's limit and,
where the extension is installed, against the extension itself.
"""

import asyncio
import io
import sys
import types
import wave
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import msgspec
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sie_server.api import openai_audio
from sie_server.api.extract import router as extract_router
from sie_server.config.model import ExtractTask, InputModalities, ModelConfig, ProfileConfig, Tasks
from sie_server.core.inference_output import ExtractOutput
from sie_server.core.preprocessor.audio import AudioPreprocessor
from sie_server.core.preprocessor_registry import PreprocessorRegistry
from sie_server.core.registry import ModelRegistry
from sie_server.core.timing import RequestTiming
from sie_server.core.worker import WorkerResult

MODEL = "nvidia/parakeet-tdt-0.6b-v3"
MAX_AUDIO_MS = 12 * 60 * 1_000
SAMPLE_RATE = 8_000
FRAMES_AT_LIMIT = MAX_AUDIO_MS * SAMPLE_RATE // 1_000
REFUSAL = f"decoded audio exceeds {MAX_AUDIO_MS} ms"


def _wav(frames: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(bytes(2 * frames))
    return buffer.getvalue()


def _stand_in_decode_audio(data: bytes, format: str | None = None) -> dict[str, Any]:
    with wave.open(io.BytesIO(data)) as wav:
        sample_rate, frames = wav.getframerate(), wav.getnframes()
    duration_ms = (frames * 1_000 + sample_rate - 1) // sample_rate
    if duration_ms > MAX_AUDIO_MS:
        raise ValueError(REFUSAL)
    sample_count = frames * 16_000 // sample_rate
    return {
        "encoding": "pcm_s16le",
        "pcm_s16le": bytes(2 * sample_count),
        "sample_rate": 16_000,
        "sample_count": sample_count,
        "duration_ms": duration_ms,
        "source_sample_rate": sample_rate,
        "source_sample_count": frames,
        "source_channels": 1,
        "container": "wav",
    }


@pytest.fixture(params=["stand-in", "compiled"])
def audio_extension(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.param == "compiled":
        pytest.importorskip("sie_audio_prep", reason="the compiled sie_audio_prep extension is not installed")
        return
    monkeypatch.setitem(sys.modules, "sie_audio_prep", types.SimpleNamespace(decode_audio=_stand_in_decode_audio))


@pytest.fixture
def worker() -> MagicMock:
    output = ExtractOutput(
        entities=[[]],
        data=[{"text": "hello world", "language": None, "duration_ms": MAX_AUDIO_MS}],
        batch_size=1,
    )

    async def submit_extract(**kwargs: Any) -> asyncio.Future[WorkerResult]:
        future: asyncio.Future[WorkerResult] = asyncio.get_running_loop().create_future()
        future.set_result(WorkerResult(output=output, timing=RequestTiming()))
        return future

    worker = MagicMock()
    worker.submit_extract = AsyncMock(side_effect=submit_extract)
    return worker


@pytest.fixture
def registry(worker: MagicMock) -> Iterator[MagicMock]:
    preprocessors = PreprocessorRegistry(max_workers=1)
    preprocessors.register(MODEL, AudioPreprocessor())
    registry = MagicMock(spec=ModelRegistry)
    registry.has_model.return_value = True
    registry.is_loaded.return_value = True
    registry.is_loading.return_value = False
    registry.is_unloading.return_value = False
    registry.is_failed.return_value = False
    registry.get_failure.return_value = None
    registry.get_worker.return_value = None
    registry.get_config.return_value = ModelConfig(
        sie_id=MODEL,
        hf_id=MODEL,
        inputs=InputModalities(text=False, audio=True),
        tasks=Tasks(extract=ExtractTask()),
        profiles={
            "default": ProfileConfig(
                adapter_path="sie_server.adapters.parakeet.adapter:ParakeetTDTAdapter",
                max_batch_tokens=MAX_AUDIO_MS,
            )
        },
    )
    registry.device = "cpu"
    registry.engine_config = None
    registry.preprocessor_registry = preprocessors
    registry.start_worker = AsyncMock(return_value=worker)
    yield registry
    preprocessors.shutdown()


@pytest.fixture
def client(registry: MagicMock, audio_extension: None) -> TestClient:
    app = FastAPI()
    app.include_router(extract_router)
    app.include_router(openai_audio.router)
    app.state.registry = registry
    return TestClient(app)


def _post_extract(client: TestClient, audio: bytes) -> Any:
    return client.post(
        f"/v1/extract/{MODEL}",
        content=msgspec.msgpack.encode({"items": [{"audio": {"data": audio, "format": "wav"}}]}),
        headers={"Content-Type": "application/msgpack", "Accept": "application/json"},
    )


def _post_transcription(client: TestClient, audio: bytes) -> Any:
    return client.post(
        "/v1/audio/transcriptions",
        data={"model": MODEL},
        files={"file": ("clip.wav", audio, "application/octet-stream")},
    )


def _prepared_payload(worker: MagicMock) -> Any:
    worker.submit_extract.assert_awaited_once()
    (prepared,) = worker.submit_extract.await_args.kwargs["prepared_items"]
    return prepared.payload


class TestExtractRoute:
    def test_audio_at_the_limit_reaches_the_worker(self, client: TestClient, worker: MagicMock) -> None:
        response = _post_extract(client, _wav(FRAMES_AT_LIMIT))

        assert response.status_code == 200
        payload = _prepared_payload(worker)
        assert payload.duration_ms == MAX_AUDIO_MS
        assert payload.source_sample_count == FRAMES_AT_LIMIT

    def test_audio_one_sample_over_the_limit_is_refused_before_the_worker(
        self, client: TestClient, worker: MagicMock
    ) -> None:
        response = _post_extract(client, _wav(FRAMES_AT_LIMIT + 1))

        assert response.status_code == 400
        assert response.json() == {"detail": {"code": "INVALID_INPUT", "message": REFUSAL}}
        worker.submit_extract.assert_not_awaited()


class TestTranscriptionRoute:
    def test_audio_at_the_limit_reaches_the_worker(self, client: TestClient, worker: MagicMock) -> None:
        response = _post_transcription(client, _wav(FRAMES_AT_LIMIT))

        assert response.status_code == 200
        payload = _prepared_payload(worker)
        assert payload.duration_ms == MAX_AUDIO_MS
        assert payload.source_sample_count == FRAMES_AT_LIMIT

    def test_audio_one_sample_over_the_limit_is_refused_before_the_worker(
        self, client: TestClient, worker: MagicMock
    ) -> None:
        response = _post_transcription(client, _wav(FRAMES_AT_LIMIT + 1))

        assert response.status_code == 400
        assert response.json() == {
            "error": {
                "message": REFUSAL,
                "type": "invalid_request_error",
                "param": None,
                "code": "invalid_request",
            }
        }
        worker.submit_extract.assert_not_awaited()
