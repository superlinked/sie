"""CPU contract fixtures; pretrained assets and target model execution are mocked."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from sie_server.adapters.qwen3_asr.adapter import Qwen3ASRAdapter, _termination
from sie_server.core.prepared import AudioPayload, PreparedItem
from sie_server.core.preprocessor.audio import AudioPreprocessor
from sie_server.types.inputs import Item
from transformers.feature_extraction_utils import BatchFeature

MODEL = "Qwen/Qwen3-ASR-1.7B-hf"
REVISION = "bcd2b5b7f32b480ab5790554cfa8347f246a14f3"


def _payload(duration_ms: int = 1000) -> AudioPayload:
    count = duration_ms * 16
    pcm = np.arange(count, dtype=np.int16).tobytes()
    return AudioPayload(
        pcm_s16le=pcm,
        sample_rate=16000,
        sample_count=count,
        duration_ms=duration_ms,
        source_sample_rate=16000,
        source_sample_count=count,
        source_channels=1,
        container="wav",
    )


def _prepared(payloads: list[AudioPayload]) -> list[PreparedItem[AudioPayload]]:
    return [
        PreparedItem(payload=payload, cost=payload.duration_cost_ms, original_index=index)
        for index, payload in enumerate(payloads)
    ]


def _loaded(*, batch_size: int = 4, maximum: int = 512, completion: list[int] | None = None):
    adapter = Qwen3ASRAdapter(MODEL, inference_batch_size=batch_size, max_new_tokens=maximum)
    adapter._model = MagicMock()
    adapter._model.dtype = torch.float32
    adapter._model.generation_config.eos_token_id = [99, 100]
    adapter._processor = MagicMock()
    adapter._preprocessor = AudioPreprocessor()
    adapter._device = "cpu"
    adapter._context_limit = 65536

    def prepare(*, audio, **kwargs):
        n = len(audio)
        return BatchFeature(
            {
                "input_ids": torch.tensor([[0, 7, 8]] * n, dtype=torch.long),
                "attention_mask": torch.tensor([[0, 1, 1]] * n, dtype=torch.long),
                "input_features": torch.ones(n, 128, 5),
                "input_features_mask": torch.ones(n, 5, dtype=torch.long),
            }
        )

    def generate(*, input_ids, **kwargs):
        suffix = torch.tensor([completion or [9, 99]] * input_ids.shape[0], dtype=torch.long)
        return SimpleNamespace(sequences=torch.cat((input_ids, suffix), dim=1))

    def decode(ids, *, return_format):
        assert return_format == "parsed"
        assert ids.shape[1] == len(completion or [9, 99])
        assert ids[0].tolist() == (completion or [9, 99])
        return [{"language": "English", "transcription": f"transcript {index}"} for index in range(ids.shape[0])]

    adapter._processor.apply_transcription_request.side_effect = prepare
    adapter._model.generate.side_effect = generate
    adapter._processor.decode.side_effect = decode
    return adapter


def test_audio_spec_and_loaded_prepared_contract() -> None:
    adapter = Qwen3ASRAdapter(MODEL)
    assert adapter.capabilities.inputs == ["audio"]
    assert adapter.capabilities.outputs == ["json"]
    with pytest.raises(NotImplementedError, match=r"Use extract\(\)"):
        adapter.encode([Item()], ["dense"])
    with pytest.raises(RuntimeError, match="Model not loaded"):
        adapter.extract([Item()])
    adapter = _loaded()
    with pytest.raises(ValueError, match="Rust-prepared"):
        adapter.extract([Item()])
    with pytest.raises(TypeError, match="non-audio"):
        adapter.extract([Item()], prepared_items=[SimpleNamespace(payload=None)])


@pytest.mark.parametrize(
    ("device", "precision", "dtype"),
    [("cpu", "bfloat16", torch.float32), ("cuda:0", "bfloat16", torch.bfloat16), ("cuda:0", "float16", torch.float16)],
)
def test_load_uses_native_auto_api_revision_and_dtype(monkeypatch, device, precision, dtype) -> None:
    processor = SimpleNamespace(feature_extractor=SimpleNamespace(sampling_rate=16000))
    model = MagicMock()
    model.config.text_config.max_position_embeddings = 65536
    processor_loader = MagicMock(return_value=processor)
    model_loader = MagicMock(return_value=model)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoProcessor=SimpleNamespace(from_pretrained=processor_loader),
            AutoModelForMultimodalLM=SimpleNamespace(from_pretrained=model_loader),
        ),
    )
    adapter = Qwen3ASRAdapter(MODEL, revision=REVISION, compute_precision=precision)
    adapter.load(device)
    processor_loader.assert_called_once_with(MODEL, revision=REVISION, trust_remote_code=False)
    model_loader.assert_called_once_with(
        MODEL, revision=REVISION, trust_remote_code=False, dtype=dtype, use_safetensors=True, attn_implementation="sdpa"
    )
    model.to.assert_called_once_with(device)
    model.eval.assert_called_once_with()
    assert isinstance(adapter.get_preprocessor(), AudioPreprocessor)
    assert adapter._context_limit == 65536


def test_missing_optional_model_api_fails_before_asset_load(monkeypatch) -> None:
    processor_loader = MagicMock()
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoProcessor=SimpleNamespace(from_pretrained=processor_loader)),
    )
    with pytest.raises(RuntimeError, match="Transformers5 bundle"):
        Qwen3ASRAdapter(MODEL, revision=REVISION).load("cpu")
    processor_loader.assert_not_called()


def test_full_mixed_duration_waveforms_context_language_and_batch_order() -> None:
    adapter = _loaded(batch_size=2)
    payloads = [_payload(500), _payload(31001), _payload(750)]
    output = adapter.extract(
        [Item()] * 3,
        prepared_items=_prepared(payloads),
        instruction="Vocabulary: Quilter",
        options={"language": "en", "max_new_tokens": 448},
    )
    assert output.entities == [[], [], []]
    assert output.batch_size == 3
    assert [row["duration_ms"] for row in output.data] == [500, 31001, 750]
    assert [row["text"] for row in output.data] == ["transcript 0", "transcript 1", "transcript 0"]
    calls = adapter._processor.apply_transcription_request.call_args_list
    assert [len(call.kwargs["audio"]) for call in calls] == [2, 1]
    for call, batch in zip(calls, [payloads[:2], payloads[2:]], strict=True):
        for waveform, payload in zip(call.kwargs["audio"], batch, strict=True):
            assert waveform.dtype == np.float32
            assert len(waveform) == payload.sample_count
            np.testing.assert_array_equal(waveform, np.frombuffer(payload.pcm_s16le, dtype="<i2") / 32768)
        assert call.kwargs["language"] == "en"
        assert call.kwargs["prompt"] == "Vocabulary: Quilter"
        assert call.kwargs["audio_kwargs"] == {
            "sampling_rate": 16000,
            "padding": True,
            "truncation": False,
            "return_attention_mask": True,
        }
        assert call.kwargs["text_kwargs"] == {"padding": True, "padding_side": "left"}
    assert all(
        call.kwargs["max_new_tokens"] == 448 and call.kwargs["do_sample"] is False
        for call in adapter._model.generate.call_args_list
    )
    assert all(row["finish_reason"] == "stop" and row["output_tokens"] == 2 for row in output.data)
    assert all("words" not in row and "segments" not in row for row in output.data)


def test_batchfeature_move_preserves_integer_ids_and_masks() -> None:
    adapter = _loaded()
    adapter._model.dtype = torch.bfloat16
    adapter.extract([Item()], prepared_items=_prepared([_payload()]))
    generated = adapter._model.generate.call_args.kwargs
    assert generated["input_features"].dtype == torch.bfloat16
    for key in ("input_ids", "attention_mask", "input_features_mask"):
        assert generated[key].dtype == torch.long
    assert generated["input_ids"].tolist() == [[0, 7, 8]]
    assert generated["attention_mask"].tolist() == [[0, 1, 1]]


def test_default_cap_and_explicit_sampling() -> None:
    adapter = _loaded()
    adapter.extract([Item()], prepared_items=_prepared([_payload()]))
    assert adapter._model.generate.call_args.kwargs["max_new_tokens"] == 512
    assert adapter._model.generate.call_args.kwargs["do_sample"] is False
    assert "temperature" not in adapter._model.generate.call_args.kwargs
    adapter.extract([Item()], prepared_items=_prepared([_payload()]), options={"temperature": 0.3})
    assert adapter._model.generate.call_args.kwargs["temperature"] == 0.3
    assert adapter._model.generate.call_args.kwargs["do_sample"] is True


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        ([1, 99, 100], (2, "stop", "observed_stop")),
        ([1, 2, 99], (3, "stop", "observed_stop")),
        ([1, 2, 3], (3, "length", "capped_as_returned")),
        ([1, 2], (2, None, "unknown")),
    ],
)
def test_observed_eos_cap_and_unknown_termination(tokens, expected) -> None:
    assert _termination(tokens, [99, 100], 3) == expected


def test_capped_transcription_is_preserved_as_returned() -> None:
    adapter = _loaded(maximum=3, completion=[1, 2, 3])
    output = adapter.extract([Item()], prepared_items=_prepared([_payload()]))
    assert output.data[0]["text"] == "transcript 0"
    assert output.data[0]["finish_reason"] == "length"
    assert output.data[0]["cap_state"] == "capped_as_returned"


def test_actual_padded_prompt_and_output_context_reject_before_generate() -> None:
    adapter = _loaded()
    adapter._context_limit = 450
    with pytest.raises(ValueError, match="prompt, batch padding"):
        adapter.extract([Item()], prepared_items=_prepared([_payload()]), options={"max_new_tokens": 448})
    adapter._model.generate.assert_not_called()
    adapter._processor.decode.assert_not_called()


def test_late_microbatch_context_overflow_rejects_entire_request_before_generate() -> None:
    adapter = _loaded()
    adapter._context_limit = 515
    adapter._processor.apply_transcription_request.side_effect = [
        BatchFeature(
            {"input_ids": torch.ones(4, 3, dtype=torch.long), "attention_mask": torch.ones(4, 3, dtype=torch.long)}
        ),
        BatchFeature(
            {"input_ids": torch.ones(2, 4, dtype=torch.long), "attention_mask": torch.ones(2, 4, dtype=torch.long)}
        ),
    ]
    with pytest.raises(ValueError, match="prompt, batch padding"):
        adapter.extract([Item()] * 6, prepared_items=_prepared([_payload()] * 6))
    assert adapter._processor.apply_transcription_request.call_count == 2
    adapter._model.generate.assert_not_called()
    adapter._processor.decode.assert_not_called()


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"timestamp_granularities": ["word"]}, "separate forced aligner"),
        ({"timestamp_granularities": ["segment"]}, "separate forced aligner"),
        ({"timestamp_granularities": "word"}, "separate forced aligner"),
        ({"language": ""}, "language must be"),
        ({"max_new_tokens": True}, "max_new_tokens must be"),
        ({"max_new_tokens": 0}, "max_new_tokens must be"),
        ({"temperature": True}, "temperature must be"),
        ({"temperature": float("nan")}, "temperature must be"),
        ({"temperature": 1.1}, "temperature must be"),
        ({"unknown": 1}, "unsupported Qwen3-ASR options"),
    ],
)
def test_invalid_options_never_run_processor_or_model(options: dict[str, Any], message: str) -> None:
    adapter = _loaded()
    with pytest.raises(ValueError, match=message):
        adapter.extract([Item()], prepared_items=_prepared([_payload()]), options=options)
    adapter._processor.apply_transcription_request.assert_not_called()
    adapter._model.generate.assert_not_called()


@pytest.mark.parametrize("kwargs", [{"labels": ["person"]}, {"output_schema": {"type": "object"}}])
def test_labels_schema_refused_before_model(kwargs) -> None:
    adapter = _loaded()
    with pytest.raises(ValueError, match="labels or output_schema"):
        adapter.extract([Item()], prepared_items=_prepared([_payload()]), **kwargs)
    adapter._model.generate.assert_not_called()


def test_misaligned_generated_or_decoded_batch_is_operational_error() -> None:
    adapter = _loaded()
    adapter._model.generate.side_effect = None
    adapter._model.generate.return_value = SimpleNamespace(sequences=torch.zeros(0, 5, dtype=torch.long))
    with pytest.raises(RuntimeError, match="generated batch"):
        adapter.extract([Item()], prepared_items=_prepared([_payload()]))
    adapter = _loaded()
    adapter._processor.decode.side_effect = None
    adapter._processor.decode.return_value = []
    with pytest.raises(RuntimeError, match="transcription batch"):
        adapter.extract([Item()], prepared_items=_prepared([_payload()]))


def test_unload_clears_loaded_state() -> None:
    adapter = _loaded()
    adapter.unload()
    assert adapter._model is None
    assert adapter._processor is None
    assert adapter._preprocessor is None
    assert adapter._context_limit is None
    assert adapter._device is None
