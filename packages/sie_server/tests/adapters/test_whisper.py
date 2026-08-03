from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from sie_server.adapters.whisper.adapter import WhisperAdapter
from sie_server.core.prepared import AudioPayload, PreparedItem
from sie_server.core.preprocessor.audio import AudioPreprocessor
from sie_server.types.inputs import Item


def _payload(*, duration_ms: int = 1_000) -> AudioPayload:
    sample_count = duration_ms * 16
    return AudioPayload(
        pcm_s16le=b"\x00\x00" * sample_count,
        sample_rate=16_000,
        sample_count=sample_count,
        duration_ms=duration_ms,
        source_sample_rate=44_100,
        source_sample_count=duration_ms * 44 + duration_ms // 10,
        source_channels=2,
        container="wav",
    )


def _loaded_adapter(output: Any) -> tuple[WhisperAdapter, MagicMock]:
    pipeline = MagicMock(return_value=output)
    processor = MagicMock()
    processor.get_prompt_ids.return_value = torch.tensor([1, 2], dtype=torch.long)
    processor.feature_extractor.n_samples = 480_000
    adapter = WhisperAdapter("openai/whisper-large-v3-turbo", pipeline_batch_size=4)
    adapter._model = MagicMock()
    adapter._processor = processor
    adapter._pipeline = pipeline
    adapter._preprocessor = AudioPreprocessor()
    adapter._device = "cpu"
    return adapter, pipeline


def test_capabilities_and_encode_contract() -> None:
    adapter = WhisperAdapter("openai/whisper-large-v3-turbo")

    assert adapter.capabilities.inputs == ["audio"]
    assert adapter.capabilities.outputs == ["json"]
    assert adapter.dims.dense is None
    with pytest.raises(NotImplementedError, match=r"Use extract\(\) instead"):
        adapter.encode([Item(text="hello")], ["dense"])


def test_extract_requires_loaded_prepared_audio() -> None:
    adapter = WhisperAdapter("openai/whisper-large-v3-turbo")
    with pytest.raises(RuntimeError, match="Model not loaded"):
        adapter.extract([Item()])

    adapter, _ = _loaded_adapter({"text": "hello", "chunks": []})
    with pytest.raises(ValueError, match="Rust-prepared audio"):
        adapter.extract([Item()])
    with pytest.raises(TypeError, match="non-audio prepared item"):
        adapter.extract(
            [Item()],
            prepared_items=[SimpleNamespace(payload=object())],
        )


def test_extract_batches_pcm_and_maps_word_and_segment_timestamps() -> None:
    adapter, pipeline = _loaded_adapter(
        {
            "text": " Hello world.",
            "chunks": [
                {"text": " Hello", "timestamp": (0.0, 0.4), "language": "english"},
                {"text": " world.", "timestamp": (0.4, 0.9), "language": "english"},
            ],
        }
    )
    payload = _payload()

    output = adapter.extract(
        [Item()],
        instruction="domain vocabulary",
        options={
            "language": "en",
            "temperature": 0.0,
            "timestamp_granularities": ["word", "segment"],
        },
        prepared_items=[PreparedItem(payload=payload, cost=1, original_index=0)],
    )

    assert output.batch_size == 1
    assert output.entities == [[]]
    assert output.data == [
        {
            "text": " Hello world.",
            "language": "english",
            "duration_ms": 1_000,
            "words": [
                {"word": " Hello", "start": 0.0, "end": 0.4},
                {"word": " world.", "start": 0.4, "end": 0.9},
            ],
            "segments": [
                {"id": 0, "start": 0.0, "end": 0.9, "text": " Hello world."},
            ],
        }
    ]
    inputs = pipeline.call_args.args[0]
    assert len(inputs) == 1
    assert inputs[0]["sampling_rate"] == 16_000
    assert inputs[0]["raw"].dtype == np.float32
    assert np.count_nonzero(inputs[0]["raw"]) == 0
    call_kwargs = pipeline.call_args.kwargs
    prompt_ids = call_kwargs["generate_kwargs"].pop("prompt_ids")
    assert isinstance(prompt_ids, torch.Tensor)
    assert prompt_ids.device == torch.device("cpu")
    assert torch.equal(prompt_ids, torch.tensor([1, 2]))
    assert call_kwargs == {
        "batch_size": 4,
        "return_timestamps": "word",
        "return_language": True,
        "generate_kwargs": {
            "language": "en",
            "temperature": 0.0,
        },
    }


def test_extract_moves_prompt_ids_to_inference_device() -> None:
    adapter, pipeline = _loaded_adapter({"text": "hello", "chunks": []})
    adapter._device = "meta"

    adapter.extract(
        [Item()],
        instruction="domain vocabulary",
        prepared_items=[PreparedItem(payload=_payload(), cost=1, original_index=0)],
    )

    prompt_ids = pipeline.call_args.kwargs["generate_kwargs"]["prompt_ids"]
    assert isinstance(prompt_ids, torch.Tensor)
    assert prompt_ids.device == torch.device("meta")


def test_extract_separates_short_and_long_audio_pipeline_batches() -> None:
    adapter, pipeline = _loaded_adapter(
        [
            {"text": "short", "chunks": []},
            {"text": "long", "chunks": []},
        ]
    )
    short = _payload(duration_ms=1_000)
    long = _payload(duration_ms=30_001)
    pipeline.side_effect = [
        [{"text": "short", "chunks": []}],
        [{"text": "long", "chunks": []}],
    ]

    output = adapter.extract(
        [Item(), Item()],
        prepared_items=[
            PreparedItem(payload=short, cost=short.duration_ms, original_index=0),
            PreparedItem(payload=long, cost=long.duration_ms, original_index=1),
        ],
    )

    assert [result["text"] for result in output.data] == ["short", "long"]
    assert pipeline.call_count == 2
    assert len(pipeline.call_args_list[0].args[0][0]["raw"]) == short.sample_count
    assert len(pipeline.call_args_list[1].args[0][0]["raw"]) == long.sample_count
    assert pipeline.call_args_list[0].kwargs["return_timestamps"] is False
    assert pipeline.call_args_list[1].kwargs["return_timestamps"] is True


def test_extract_requests_segment_timestamps_for_short_audio() -> None:
    adapter, pipeline = _loaded_adapter(
        {
            "text": "hello",
            "chunks": [{"text": "hello", "timestamp": (0.0, 0.9), "language": "english"}],
        }
    )

    output = adapter.extract(
        [Item()],
        options={"timestamp_granularities": ["segment"]},
        prepared_items=[PreparedItem(payload=_payload(), cost=1, original_index=0)],
    )

    assert pipeline.call_args.kwargs["return_timestamps"] is True
    assert output.data[0]["segments"] == [{"id": 0, "start": 0.0, "end": 0.9, "text": "hello"}]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"unknown": True}, "unsupported Whisper options"),
        ({"language": ""}, "language must be"),
        ({"temperature": True}, "temperature must be"),
        ({"temperature": 1.1}, "temperature must be"),
        ({"timestamp_granularities": "word"}, "must be a list"),
        ({"timestamp_granularities": ["token"]}, "unsupported timestamp"),
    ],
)
def test_extract_rejects_invalid_options(options: dict[str, Any], message: str) -> None:
    adapter, _ = _loaded_adapter({"text": "hello", "chunks": []})

    with pytest.raises(ValueError, match=message):
        adapter.extract(
            [Item()],
            options=options,
            prepared_items=[PreparedItem(payload=_payload(), cost=1, original_index=0)],
        )


def test_extract_rejects_labels_schema_and_misaligned_pipeline() -> None:
    prepared = [PreparedItem(payload=_payload(), cost=1, original_index=0)]
    adapter, _ = _loaded_adapter([])

    with pytest.raises(ValueError, match="entity labels"):
        adapter.extract([Item()], labels=["person"], prepared_items=prepared)
    with pytest.raises(ValueError, match="output_schema"):
        adapter.extract([Item()], output_schema={"type": "object"}, prepared_items=prepared)
    with pytest.raises(RuntimeError, match="misaligned batch"):
        adapter.extract([Item()], prepared_items=prepared)


def test_direct_preprocessor_uses_rust_and_strips_encoded_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    decode_audio = MagicMock(
        return_value={
            "pcm_s16le": b"\x00\x00" * 17,
            "sample_rate": 16_000,
            "sample_count": 17,
            "duration_ms": 2,
            "source_sample_rate": 44_100,
            "source_sample_count": 45,
            "source_channels": 2,
            "container": "wav",
        }
    )
    rust_extension = SimpleNamespace(decode_audio=decode_audio)
    monkeypatch.setattr(
        "sie_server.core.preprocessor.audio.importlib.import_module",
        lambda name: rust_extension,
    )
    item = Item(audio={"data": b"encoded", "format": "wav", "sample_rate": 44_100})

    batch = AudioPreprocessor().prepare([item], config=MagicMock())

    decode_audio.assert_called_once_with(b"encoded", "wav")
    assert item.audio == {"data": b"encoded", "format": "wav", "sample_rate": 44_100}
    assert batch.modality == "audio"
    assert batch.total_cost == 2
    assert batch.items[0].payload.duration_ms == 2


def test_direct_preprocessor_reports_missing_audio_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_extension(name: str) -> None:
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(
        "sie_server.core.preprocessor.audio.importlib.import_module",
        missing_extension,
    )

    with pytest.raises(RuntimeError, match="SIE image or workspace build"):
        AudioPreprocessor().prepare(
            [Item(audio={"data": b"encoded", "format": "wav"})],
            config=MagicMock(),
        )


def test_direct_preprocessor_rejects_declared_sample_rate_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rust_extension = SimpleNamespace(
        decode_audio=lambda data, fmt: {
            "pcm_s16le": b"\x00\x00" * 16,
            "sample_rate": 16_000,
            "sample_count": 16,
            "duration_ms": 1,
            "source_sample_rate": 48_000,
            "source_channels": 1,
            "source_sample_count": 48,
            "container": "wav",
        }
    )
    monkeypatch.setattr(
        "sie_server.core.preprocessor.audio.importlib.import_module",
        lambda name: rust_extension,
    )

    with pytest.raises(ValueError, match="does not match decoded"):
        AudioPreprocessor().prepare(
            [Item(audio={"data": b"encoded", "sample_rate": 44_100})],
            config=MagicMock(),
        )
