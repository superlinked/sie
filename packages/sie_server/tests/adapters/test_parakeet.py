from __future__ import annotations

import dataclasses
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import yaml
from packaging.specifiers import SpecifierSet
from sie_sdk.bundle_utils import match_bundle_models
from sie_server.adapters.parakeet import adapter as parakeet_adapter
from sie_server.adapters.parakeet.adapter import (
    _EMPTY_RETRY_MAX_S,
    SUPPORTED_LANGUAGES,
    ParakeetTDTAdapter,
    _length_chunks,
    _segments_from_words,
    _Transcript,
    _words_from_tokens,
)
from sie_server.adapters.parakeet.decoding import TdtHypothesis, greedy_tdt_decode
from sie_server.bundle_requirements import resolve_bundle_requirements
from sie_server.core.loader import load_model_configs, reject_unknown_loadtime_options, resolve_adapter_class
from sie_server.core.prepared import AudioPayload, PreparedItem
from sie_server.core.preprocessor.audio import AudioPreprocessor
from sie_server.types.inputs import Item

SERVER_ROOT = Path(__file__).resolve().parents[2]
MODELS = SERVER_ROOT / "models"
BUNDLES = SERVER_ROOT / "bundles"
MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
BLANK = 9
VOCAB = 10
DURATIONS = (0, 1, 2, 3, 4)


# ---------------------------------------------------------------------------
# Deterministic prediction and joint networks for the greedy loop.
#
# The prediction network's state is a counter of how many times it was
# updated (1 after the start step), carried in feature 0. Encoder frame
# features carry ``row * 1000 + frame`` in feature 1, so the joint's input
# ``frame + prediction`` tells a scripted head which row, frame and decoder
# state it is scoring.
# ---------------------------------------------------------------------------


class _CountingLSTM(torch.nn.Module):
    num_layers = 1
    hidden_size = 2

    def forward(
        self, inputs: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hidden, cell = state
        updated = hidden + torch.tensor([1.0, 0.0])
        return updated[-1].unsqueeze(1), (updated, cell)


class _FakeDecoder:
    def __init__(self) -> None:
        self.lstm = _CountingLSTM()
        self.fed: list[list[int]] = []

    def embedding(self, ids: torch.Tensor) -> torch.Tensor:
        self.fed.append(ids[:, 0].tolist())
        return torch.zeros(ids.shape[0], 1, 2)

    def decoder_projector(self, output: torch.Tensor) -> torch.Tensor:
        return output


class _ScriptedJoint:
    """Emit ``script(row, frame, updates) -> (token, duration_index)`` as one-hot logits."""

    def __init__(self, script: Any, *, vocab_size: int = VOCAB, num_durations: int = len(DURATIONS)) -> None:
        self.script = script
        self.vocab_size = vocab_size
        self.num_durations = num_durations
        self.seen: list[tuple[int, int, int]] = []

    def activation(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def head(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(x.shape[0], self.vocab_size + self.num_durations)
        for index, (updates, code) in enumerate(x.tolist()):
            row, frame = divmod(round(code), 1000)
            self.seen.append((row, frame, round(updates)))
            token, duration_index = self.script(row, frame, round(updates))
            logits[index, token] = 1.0
            logits[index, self.vocab_size + duration_index] = 1.0
        return logits


def _encoder(frames: int, rows: int = 1) -> torch.Tensor:
    codes = torch.tensor([[row * 1000 + frame for frame in range(frames)] for row in range(rows)], dtype=torch.float32)
    return torch.stack([torch.zeros_like(codes), codes], dim=-1)


def _decode(
    script: Any, lengths: list[int], *, frames: int | None = None, durations: tuple[int, ...] = DURATIONS
) -> tuple[list[TdtHypothesis], _FakeDecoder, _ScriptedJoint]:
    decoder = _FakeDecoder()
    joint = _ScriptedJoint(script, num_durations=len(durations))
    hypotheses = greedy_tdt_decode(
        decoder,
        joint,
        _encoder(frames or max(lengths), len(lengths)),
        torch.tensor(lengths),
        blank_id=BLANK,
        vocab_size=VOCAB,
        durations=durations,
    )
    return hypotheses, decoder, joint


def test_greedy_decode_follows_tokens_durations_and_blanks() -> None:
    script = {
        (0, 1): (5, 0),  # emit, stay on frame 0
        (0, 2): (6, 2),  # emit, advance 2
        (2, 3): (BLANK, 0),  # blank with duration 0 still advances one frame
        (3, 3): (7, 1),
    }
    (hypothesis,), decoder, _ = _decode(lambda row, frame, updates: script[(frame, updates)], [4])

    assert hypothesis.token_ids == [5, 6, 7]
    assert hypothesis.step_ids == [5, 6, BLANK, 7]
    assert hypothesis.step_durations == [0, 2, 1, 1]
    # The prediction network starts from blank and is fed each emitted token.
    assert decoder.fed == [[BLANK], [5], [6], [7]]


def test_greedy_decode_maps_duration_index_to_frames() -> None:
    (hypothesis,), _, _ = _decode(lambda row, frame, updates: (3, 1), [7], durations=(0, 3))

    assert hypothesis.step_durations == [3, 3, 3]
    assert hypothesis.token_ids == [3, 3, 3]


def test_greedy_decode_forces_progress_after_max_symbols_on_one_frame() -> None:
    (hypothesis,), _, _ = _decode(lambda row, frame, updates: (4, 0), [3])

    # Ten emissions per frame, the tenth forced one frame forward.
    assert hypothesis.token_ids == [4] * 30
    assert hypothesis.step_durations == ([0] * 9 + [1]) * 3


def test_greedy_decode_counts_consecutive_symbols_per_frame() -> None:
    # Nine stays, an advance by 1, then nine more stays: the counter resets on advance.
    def script(row: int, frame: int, updates: int) -> tuple[int, int]:
        return (4, 1) if updates == 10 else (4, 0)

    (hypothesis,), _, _ = _decode(script, [2])

    assert hypothesis.step_durations[:10] == [0] * 9 + [1]
    assert hypothesis.step_durations[10:] == [0] * 9 + [1]


def test_greedy_decode_stops_each_row_at_its_own_length() -> None:
    def script(row: int, frame: int, updates: int) -> tuple[int, int]:
        if row == 0:
            return (8, 1) if frame >= 2 else (2, 1)  # frames >= 2 are padding for row 0
        return (3, 1)

    hypotheses, _, _ = _decode(script, [2, 5])

    assert hypotheses[0].token_ids == [2, 2]
    assert hypotheses[1].token_ids == [3] * 5
    assert sum(hypotheses[0].step_durations) == 2
    assert sum(hypotheses[1].step_durations) == 5


def test_greedy_decode_updates_prediction_state_only_for_emitting_rows() -> None:
    def script(row: int, frame: int, updates: int) -> tuple[int, int]:
        if row == 0 and frame == 0:
            return (5, 1)
        return (BLANK, 1)

    hypotheses, decoder, joint = _decode(script, [2, 2])

    assert [hypothesis.token_ids for hypothesis in hypotheses] == [[5], []]
    # Second step: row 0 scored with its updated state, row 1 with the start state.
    assert (0, 1, 2) in joint.seen
    assert (1, 1, 1) in joint.seen
    assert decoder.fed == [[BLANK, BLANK], [5, BLANK]]


def test_greedy_decode_handles_empty_input_and_rejects_invalid_guard() -> None:
    assert (
        greedy_tdt_decode(
            _FakeDecoder(),
            _ScriptedJoint(lambda *_: (BLANK, 1)),
            torch.zeros(0, 3, 2),
            torch.zeros(0, dtype=torch.long),
            blank_id=BLANK,
            vocab_size=VOCAB,
            durations=DURATIONS,
        )
        == []
    )
    (hypothesis,), _, _ = _decode(lambda *_: (5, 1), [0], frames=3)
    assert hypothesis == TdtHypothesis()
    with pytest.raises(ValueError, match="max_symbols_per_frame"):
        greedy_tdt_decode(
            _FakeDecoder(),
            _ScriptedJoint(lambda *_: (BLANK, 1)),
            _encoder(2),
            torch.tensor([2]),
            blank_id=BLANK,
            vocab_size=VOCAB,
            durations=DURATIONS,
            max_symbols_per_frame=0,
        )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


def _payload(*, duration_ms: int = 1_000) -> AudioPayload:
    sample_count = duration_ms * 16
    return AudioPayload(
        pcm_s16le=b"\x00\x00" * sample_count,
        sample_rate=16_000,
        sample_count=sample_count,
        duration_ms=duration_ms,
        source_sample_rate=16_000,
        source_sample_count=sample_count,
        source_channels=1,
        container="wav",
    )


def _prepared(*payloads: AudioPayload) -> list[PreparedItem[AudioPayload]]:
    return [
        PreparedItem(payload=payload, cost=payload.duration_ms, original_index=index)
        for index, payload in enumerate(payloads)
    ]


def _loaded_adapter(**kwargs: Any) -> ParakeetTDTAdapter:
    adapter = ParakeetTDTAdapter(MODEL_ID, **kwargs)
    adapter._model = MagicMock()
    adapter._processor = MagicMock()
    adapter._preprocessor = AudioPreprocessor()
    adapter._device = "cpu"
    adapter._blank_id = BLANK
    adapter._vocab_size = VOCAB
    adapter._durations = DURATIONS
    return adapter


def _transcript(text: str, offset_s: float = 0.0) -> _Transcript:
    hypothesis = TdtHypothesis(token_ids=[1] if text else [], step_ids=[1] if text else [], step_durations=[1])
    return _Transcript(text=text, hypothesis=hypothesis, offset_s=offset_s)


def test_capabilities_and_encode_contract() -> None:
    adapter = ParakeetTDTAdapter(MODEL_ID)

    assert adapter.capabilities.inputs == ["audio"]
    assert adapter.capabilities.outputs == ["json"]
    assert adapter.dims.dense is None
    assert adapter.get_preprocessor() is None
    with pytest.raises(NotImplementedError, match=r"Use extract\(\) instead"):
        adapter.encode([Item(text="hello")], ["dense"])
    with pytest.raises(ValueError, match="max_padded_batch_ms"):
        ParakeetTDTAdapter(MODEL_ID, max_padded_batch_ms=0)


def test_extract_requires_loaded_prepared_audio() -> None:
    with pytest.raises(RuntimeError, match="Model not loaded"):
        ParakeetTDTAdapter(MODEL_ID).extract([Item()])

    adapter = _loaded_adapter()
    with pytest.raises(ValueError, match="Rust-prepared audio"):
        adapter.extract([Item()])
    with pytest.raises(TypeError, match="non-audio prepared item"):
        adapter.extract([Item()], prepared_items=[SimpleNamespace(payload=object())])
    resampled = dataclasses.replace(_payload(), sample_rate=8_000)
    with pytest.raises(ValueError, match="requires 16000 Hz audio"):
        adapter.extract([Item()], prepared_items=_prepared(resampled))


def test_extract_rejects_labels_and_schema() -> None:
    adapter = _loaded_adapter()
    prepared = _prepared(_payload())

    with pytest.raises(ValueError, match="entity labels"):
        adapter.extract([Item()], labels=["person"], prepared_items=prepared)
    with pytest.raises(ValueError, match="output_schema"):
        adapter.extract([Item()], output_schema={"type": "object"}, prepared_items=prepared)


@pytest.mark.parametrize(
    ("options", "instruction", "message"),
    [
        ({"unknown": True}, None, "unsupported Parakeet options: unknown"),
        ({"language": "ja"}, None, "this model transcribes 25 languages"),
        ({"language": ""}, None, "is not supported"),
        ({"language": 7}, None, "is not supported"),
        ({}, "domain vocabulary", "does not accept a prompt"),
        ({"temperature": 0.2}, None, "temperature must be 0 or omitted"),
        ({"temperature": True}, None, "temperature must be 0 or omitted"),
        ({"temperature": "0"}, None, "temperature must be 0 or omitted"),
        ({"timestamp_granularities": "word"}, None, "must be a list"),
        ({"timestamp_granularities": ["token"]}, None, "unsupported timestamp granularities: token"),
    ],
)
def test_extract_rejects_invalid_options(options: dict[str, Any], instruction: str | None, message: str) -> None:
    adapter = _loaded_adapter()
    adapter._decode = MagicMock()

    with pytest.raises(ValueError, match=message):
        adapter.extract([Item()], options=options, instruction=instruction, prepared_items=_prepared(_payload()))
    adapter._decode.assert_not_called()


def test_unsupported_language_error_lists_the_supported_codes() -> None:
    adapter = _loaded_adapter()

    with pytest.raises(ValueError, match="language") as error:
        adapter.extract([Item()], options={"language": "zh"}, prepared_items=_prepared(_payload()))
    assert ", ".join(sorted(SUPPORTED_LANGUAGES)) in str(error.value)
    assert len(SUPPORTED_LANGUAGES) == 25


@pytest.mark.parametrize(
    ("options", "instruction"),
    [
        ({"language": "en"}, None),
        ({"language": " DE "}, None),
        ({"language": "uk", "temperature": 0}, ""),
        ({"language": None, "temperature": 0.0}, "   "),
        ({"timestamp_granularities": ("segment",)}, None),
        ({"timestamp_granularities": []}, None),
        (None, None),
    ],
)
def test_extract_accepts_supported_options(options: dict[str, Any] | None, instruction: str | None) -> None:
    adapter = _loaded_adapter()
    adapter._decode = MagicMock(return_value=[_transcript("hallo")])
    adapter._processor.decode.return_value = (["hallo"], [[{"token": "hallo", "start": 0.0, "end": 0.08}]])

    output = adapter.extract([Item()], options=options, instruction=instruction, prepared_items=_prepared(_payload()))

    assert output.data is not None
    assert output.data[0]["text"] == "hallo"
    assert output.data[0]["language"] is None


def test_extract_returns_the_transcription_contract() -> None:
    adapter = _loaded_adapter()
    adapter._decode = MagicMock()
    first, second = _payload(duration_ms=1_000), _payload(duration_ms=2_000)
    adapter._decode.side_effect = [[_transcript("Hello world."), _transcript("")], [_transcript("")], [_transcript("")]]

    output = adapter.extract([Item(), Item()], prepared_items=_prepared(first, second))

    assert output.batch_size == 2
    assert output.entities == [[], []]
    # The batch is decoded longest first; the result keeps request order.
    assert output.data == [
        {"text": "", "language": None, "duration_ms": 1_000},
        {"text": "Hello world.", "language": None, "duration_ms": 2_000},
    ]


def test_transcribe_groups_rows_by_padded_length_budget() -> None:
    adapter = _loaded_adapter(max_padded_batch_ms=1_000)
    calls: list[list[int]] = []

    def fake_decode(waveforms: list[np.ndarray], *, offset_s: float = 0.0) -> list[_Transcript]:
        calls.append([waveform.shape[0] for waveform in waveforms])
        return [_transcript(f"row of {waveform.shape[0]}") for waveform in waveforms]

    adapter._decode = fake_decode
    lengths = [4_800, 9_600, 24_000, 3_200, 8_000]  # 0.3, 0.6, 1.5, 0.2, 0.5 s
    transcripts = adapter._transcribe([np.ones(length, dtype=np.float32) for length in lengths])

    # Longest first; a chunk holds rows * longest <= 1 s; an over-budget row runs alone.
    assert calls == [[24_000], [9_600], [8_000, 4_800], [3_200]]
    assert [transcript.text for transcript in transcripts] == [f"row of {length}" for length in lengths]


def test_length_chunks_never_returns_an_empty_chunk() -> None:
    assert _length_chunks([], 10) == []
    assert _length_chunks([5, 5, 5], 15) == [[0, 1, 2]]
    assert _length_chunks([5, 20, 5], 15) == [[1], [0, 2]]


def test_empty_rows_are_redecoded_alone_with_end_then_both_sides_padding() -> None:
    adapter = _loaded_adapter()
    calls: list[tuple[list[int], float]] = []
    lengths = [16_000, 8_000, 4_800, 6_400, 100]
    originals = [np.full(length, 0.5, dtype=np.float32) for length in lengths]
    # Which (original length, leading pad, trailing pad) decodes to text.
    recovers = {(16_000, 0, 0), (8_000, 0, 4_000), (4_800, 4_000, 4_000)}

    def fake_decode(waveforms: list[np.ndarray], *, offset_s: float = 0.0) -> list[_Transcript]:
        calls.append(([waveform.shape[0] for waveform in waveforms], offset_s))
        results = []
        for waveform in waveforms:
            nonzero = np.flatnonzero(waveform)
            leading, trailing = int(nonzero[0]), waveform.shape[0] - int(nonzero[-1]) - 1
            original = waveform[leading : waveform.shape[0] - trailing]
            assert np.all(waveform[:leading] == 0)
            assert np.all(original == 0.5)
            key = (original.shape[0], leading, trailing)
            results.append(_transcript(f"text {original.shape[0]}" if key in recovers else "", offset_s))
        return results

    adapter._decode = fake_decode
    transcripts = adapter._transcribe(originals)

    assert calls == [
        ([16_000, 8_000, 6_400, 4_800], 0.0),  # one batched pass; the 100-sample clip is too short to decode
        ([12_000], 0.0),  # row 1 recovers with 0.25 s appended
        ([8_800], 0.0),  # row 2: still empty with 0.25 s appended ...
        ([12_800], 0.25),  # ... recovers with 0.25 s on both sides
        ([10_400], 0.0),  # row 3 stays empty
        ([14_400], 0.25),
    ]
    assert [transcript.text for transcript in transcripts] == ["text 16000", "text 8000", "text 4800", "", ""]
    assert [transcript.offset_s for transcript in transcripts] == [0.0, 0.0, 0.25, 0.0, 0.0]


def test_empty_rows_longer_than_thirty_seconds_are_not_redecoded() -> None:
    adapter = _loaded_adapter()
    calls: list[list[int]] = []

    def fake_decode(waveforms: list[np.ndarray], *, offset_s: float = 0.0) -> list[_Transcript]:
        calls.append([waveform.shape[0] for waveform in waveforms])
        return [_transcript("", offset_s) for _ in waveforms]

    adapter._decode = fake_decode
    limit = round(_EMPTY_RETRY_MAX_S * 16_000)
    transcripts = adapter._transcribe([np.ones(limit + 1, dtype=np.float32), np.ones(limit, dtype=np.float32)])

    assert _EMPTY_RETRY_MAX_S == 30.0
    # The row just over 30 s gets only the batched pass; the 30 s row is retried twice.
    assert calls == [[limit + 1, limit], [limit + 4_000], [limit + 8_000]]
    assert [transcript.text for transcript in transcripts] == ["", ""]


def test_decode_runs_encoder_then_greedy_loop_and_processor_text(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _loaded_adapter()
    adapter._dtype = torch.float64
    features = torch.ones(2, 6, 4)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    adapter._processor.return_value = {"input_features": features, "attention_mask": mask}
    adapter._processor.batch_decode.return_value = [" Hello world. ", ""]
    encoder_output = torch.zeros(2, 3, 2)
    adapter._model.get_audio_features.return_value = SimpleNamespace(
        pooler_output=encoder_output,
        attention_mask=torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.int32),
    )
    hypotheses = [TdtHypothesis(token_ids=[1, 2]), TdtHypothesis()]
    greedy = MagicMock(return_value=hypotheses)
    monkeypatch.setattr(parakeet_adapter, "greedy_tdt_decode", greedy)
    waveforms = [np.zeros(960, dtype=np.float32), np.zeros(480, dtype=np.float32)]

    transcripts = adapter._decode(waveforms, offset_s=0.25)

    processor_kwargs = adapter._processor.call_args.kwargs
    assert processor_kwargs["audio"] is waveforms
    assert (processor_kwargs["sampling_rate"], processor_kwargs["return_tensors"]) == (16_000, "pt")
    encoder_kwargs = adapter._model.get_audio_features.call_args.kwargs
    assert encoder_kwargs["input_features"].dtype == torch.float64
    assert torch.equal(encoder_kwargs["attention_mask"], mask)
    assert encoder_kwargs["output_attention_mask"] is True
    args, kwargs = greedy.call_args
    assert args[0] is adapter._model.decoder
    assert args[1] is adapter._model.joint
    assert args[2] is encoder_output
    assert args[3].tolist() == [3, 1]
    assert kwargs == {"blank_id": BLANK, "vocab_size": VOCAB, "durations": DURATIONS}
    adapter._processor.batch_decode.assert_called_once_with([[1, 2], []], skip_special_tokens=True, group_tokens=False)
    assert [(t.text, t.hypothesis, t.offset_s) for t in transcripts] == [
        ("Hello world.", hypotheses[0], 0.25),
        ("", hypotheses[1], 0.25),
    ]


def _timed(token: str, start: float, end: float) -> dict[str, Any]:
    return {"token": token, "start": start, "end": end}


def test_word_and_segment_timestamps_come_from_processor_token_offsets() -> None:
    adapter = _loaded_adapter()
    hypothesis = TdtHypothesis(token_ids=[4, 5, 6], step_ids=[4, BLANK, 5, 6], step_durations=[1, 2, 1, 0])
    adapter._decode = MagicMock(return_value=[_Transcript(text="Hello world. Next", hypothesis=hypothesis)])
    adapter._processor.decode.return_value = (
        ["Hello world. Next"],
        [
            [
                _timed("Hel", 0.0, 0.08),
                _timed("lo", 0.08, 0.16),
                _timed(" world", 0.32, 0.48),
                _timed(".", 0.48, 0.48),
                _timed(" Next", 0.56, 0.64),
            ]
        ],
    )

    output = adapter.extract(
        [Item()],
        options={"timestamp_granularities": ["word", "segment"]},
        prepared_items=_prepared(_payload(duration_ms=1_000)),
    )

    (ids,) = adapter._processor.decode.call_args.args
    assert ids.tolist() == [[4, BLANK, 5, 6]]
    decode_kwargs = adapter._processor.decode.call_args.kwargs
    assert decode_kwargs["durations"].tolist() == [[1, 2, 1, 0]]
    assert decode_kwargs["skip_special_tokens"] is True
    assert decode_kwargs["group_tokens"] is False
    assert output.data == [
        {
            "text": "Hello world. Next",
            "language": None,
            "duration_ms": 1_000,
            "words": [
                {"word": "Hello", "start": 0.0, "end": 0.16},
                {"word": "world.", "start": 0.32, "end": 0.48},
                {"word": "Next", "start": 0.56, "end": 0.64},
            ],
            "segments": [
                {"id": 0, "start": 0.0, "end": 0.48, "text": "Hello world."},
                {"id": 1, "start": 0.56, "end": 0.64, "text": "Next"},
            ],
        }
    ]


def test_timestamps_remove_leading_padding_and_stay_within_the_recording() -> None:
    adapter = _loaded_adapter()
    hypothesis = TdtHypothesis(token_ids=[4, 5], step_ids=[4, 5], step_durations=[4, 4])
    adapter._decode = MagicMock(
        side_effect=[
            [_Transcript()],
            [_Transcript()],
            [_Transcript(text="Oh no", hypothesis=hypothesis, offset_s=0.25)],
        ]
    )
    adapter._processor.decode.return_value = (["Oh no"], [[_timed("Oh", 0.08, 0.4), _timed(" no", 0.4, 0.72)]])

    output = adapter.extract(
        [Item()],
        options={"timestamp_granularities": ["segment"]},
        prepared_items=_prepared(_payload(duration_ms=300)),
    )

    assert output.data == [
        {
            "text": "Oh no",
            "language": None,
            "duration_ms": 300,
            "segments": [{"id": 0, "start": 0.0, "end": 0.3, "text": "Oh no"}],
        }
    ]


def test_empty_transcript_has_empty_timestamps_without_decoding_offsets() -> None:
    adapter = _loaded_adapter()
    adapter._decode = MagicMock(return_value=[_Transcript()])

    output = adapter.extract(
        [Item()],
        options={"timestamp_granularities": ["word", "segment"]},
        prepared_items=_prepared(_payload()),
    )

    assert output.data == [{"text": "", "language": None, "duration_ms": 1_000, "words": [], "segments": []}]
    adapter._processor.decode.assert_not_called()


def test_words_merge_tokens_at_leading_spaces() -> None:
    words = _words_from_tokens(
        [
            _timed("¿", 0.0, 0.0),
            _timed("Qué", 0.0, 0.16),
            _timed("?", 0.16, 0.16),
            _timed(" ", 0.24, 0.32),
            _timed("Bien", 0.32, 0.4),
            _timed(" 3", 0.48, 0.56),
            _timed(".5", 0.56, 0.6400000000000001),
        ]
    )

    assert words == [
        {"word": "¿Qué?", "start": 0.0, "end": 0.16},
        {"word": "Bien", "start": 0.24, "end": 0.4},
        {"word": "3.5", "start": 0.48, "end": 0.64},
    ]


def test_segments_split_at_sentence_ends_and_after_thirty_seconds() -> None:
    def word(text: str, start: float, end: float) -> dict[str, Any]:
        return {"word": text, "start": start, "end": end}

    words = [
        word("Wait!", 0.0, 0.4),
        word('"Really?"', 0.5, 1.0),
        word("Yes…", 1.2, 1.5),
        word("long", 2.0, 20.0),
        word("talk", 20.0, 32.0),
        word("tail", 32.5, 33.0),
    ]

    assert _segments_from_words(words) == [
        {"id": 0, "start": 0.0, "end": 0.4, "text": "Wait!"},
        {"id": 1, "start": 0.5, "end": 1.0, "text": '"Really?"'},
        {"id": 2, "start": 1.2, "end": 1.5, "text": "Yes…"},
        {"id": 3, "start": 2.0, "end": 32.0, "text": "long talk"},
        {"id": 4, "start": 32.5, "end": 33.0, "text": "tail"},
    ]
    assert _segments_from_words([]) == []


def _fake_transformers(
    monkeypatch: pytest.MonkeyPatch, *, decoder_type: str | None = None, version: str = "5.18.0"
) -> tuple[MagicMock, MagicMock, MagicMock]:
    processor = MagicMock()
    processor.feature_extractor.sampling_rate = 16_000
    processor.feature_extractor.hop_length = 160
    processor.decoder_type = decoder_type
    model = MagicMock()
    model.config = SimpleNamespace(blank_token_id=8192, vocab_size=8193, durations=[0, 1, 2, 3, 4])
    auto_processor = MagicMock()
    auto_processor.from_pretrained.return_value = processor
    auto_model = MagicMock()
    auto_model.from_pretrained.return_value = model
    # Stand-in for transformers 5.x (AutoModelForTDT); the unit environment may carry 4.x.
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(__version__=version, AutoProcessor=auto_processor, AutoModelForTDT=auto_model),
    )
    return auto_processor, auto_model, model


@pytest.mark.parametrize(
    ("device", "precision", "dtype"),
    [("cpu", "bfloat16", torch.float32), ("cuda:0", "bfloat16", torch.bfloat16), ("cuda", "float16", torch.float16)],
)
def test_load_pins_revision_and_reads_the_checkpoint_layout(
    monkeypatch: pytest.MonkeyPatch, device: str, precision: str, dtype: torch.dtype
) -> None:
    auto_processor, auto_model, model = _fake_transformers(monkeypatch)
    adapter = ParakeetTDTAdapter(MODEL_ID, compute_precision=precision, revision="abc123")

    adapter.load(device)

    auto_processor.from_pretrained.assert_called_once_with(MODEL_ID, revision="abc123")
    auto_model.from_pretrained.assert_called_once_with(MODEL_ID, dtype=dtype, revision="abc123")
    model.to.assert_called_once_with(device)
    model.eval.assert_called_once_with()
    assert adapter._dtype == dtype
    assert (adapter._blank_id, adapter._vocab_size, adapter._durations) == (8192, 8193, (0, 1, 2, 3, 4))
    assert (adapter._sample_rate, adapter._min_samples) == (16_000, 320)
    # The checkpoint's processor config has no decoder type; it is pinned once.
    assert adapter._processor.decoder_type == "tdt"
    assert isinstance(adapter.get_preprocessor(), AudioPreprocessor)

    adapter.unload()
    assert adapter._model is None
    assert adapter._processor is None


def test_load_keeps_a_decoder_type_the_processor_declares(monkeypatch: pytest.MonkeyPatch) -> None:
    auto_processor, _, _ = _fake_transformers(monkeypatch, decoder_type="rnnt")
    adapter = ParakeetTDTAdapter(MODEL_ID)

    adapter.load("cpu")

    assert auto_processor.from_pretrained.return_value.decoder_type == "rnnt"


@pytest.mark.parametrize("version", ["5.14.1", "5.17.0", "5.18.0.dev0"])
def test_load_refuses_transformers_that_mask_padding_with_negative_infinity(
    monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    auto_processor, auto_model, _ = _fake_transformers(monkeypatch, version=version)
    adapter = ParakeetTDTAdapter(MODEL_ID)

    with pytest.raises(RuntimeError, match=rf"requires transformers>=5\.18, found {re.escape(version)}"):
        adapter.load("cpu")

    auto_processor.from_pretrained.assert_not_called()
    auto_model.from_pretrained.assert_not_called()
    assert adapter._model is None


@pytest.mark.parametrize("version", ["5.18.0", "5.19.0"])
def test_load_accepts_transformers_with_finite_padding_mask(monkeypatch: pytest.MonkeyPatch, version: str) -> None:
    _, auto_model, model = _fake_transformers(monkeypatch, version=version)
    adapter = ParakeetTDTAdapter(MODEL_ID)

    adapter.load("cpu")

    auto_model.from_pretrained.assert_called_once()
    assert adapter._model is model


# ---------------------------------------------------------------------------
# Catalog and bundle
# ---------------------------------------------------------------------------


def test_model_config_is_pinned_audio_extract_and_routed_by_transformers5() -> None:
    config = load_model_configs(MODELS)[MODEL_ID]

    assert config.hf_id == MODEL_ID
    assert config.hf_revision == "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
    assert config.inputs.audio is True
    assert config.inputs.text is False
    assert config.tasks.extract is not None
    assert config.tasks.encode is None
    assert config.tasks.score is None
    profile = config.resolve_profile("default")
    assert profile.adapter_path == "sie_server.adapters.parakeet.adapter:ParakeetTDTAdapter"
    assert profile.compute_precision == "bfloat16"
    # One 12-minute request (SIE's per-request audio cap) per extract call.
    assert profile.max_batch_tokens == 12 * 60 * 1_000
    adapter_class = resolve_adapter_class(config, MODELS)
    assert adapter_class is ParakeetTDTAdapter
    reject_unknown_loadtime_options(adapter_class, profile.loadtime, model_name=MODEL_ID)
    assert MODEL_ID in match_bundle_models(BUNDLES / "transformers5.yaml", MODELS)
    assert MODEL_ID not in match_bundle_models(BUNDLES / "default.yaml", MODELS)


def test_transformers5_bundle_carries_the_parakeet_runtime() -> None:
    bundle = yaml.safe_load((BUNDLES / "transformers5.yaml").read_text())
    requirements = resolve_bundle_requirements(bundle["deps"])

    assert "sie_server.adapters.parakeet.adapter" in bundle["adapters"]
    # The bundle's range admits a transformers release the adapter loads on.
    assert parakeet_adapter._MIN_TRANSFORMERS_VERSION in SpecifierSet(bundle["deps"]["transformers"])
    assert "librosa==1.0.0" in requirements
