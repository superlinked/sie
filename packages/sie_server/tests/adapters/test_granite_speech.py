from __future__ import annotations

import dataclasses
import sys
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from sie_sdk.bundle_utils import match_bundle_models
from sie_server.adapters.granite_speech import adapter as granite_adapter
from sie_server.adapters.granite_speech.adapter import (
    DEFAULT_PROMPT,
    GraniteSpeechAdapter,
    _length_chunks,
    _pause_windows,
    log_mel_features,
    mel_filterbank,
)
from sie_server.core.loader import load_model_configs, reject_unknown_loadtime_options, resolve_adapter_class
from sie_server.core.prepared import AudioPayload, PreparedItem
from sie_server.core.preprocessor.audio import AudioPreprocessor
from sie_server.types.inputs import Item

SERVER_ROOT = Path(__file__).resolve().parents[2]
MODELS = SERVER_ROOT / "models"
BUNDLES = SERVER_ROOT / "bundles"
MODEL_ID = "ibm-granite/granite-speech-5.0-2b"
SAMPLE_RATE = 16_000
AUDIO_ID = 7
EOS_ID = 2
PAD_ID = 1
HIDDEN = 4
# 100 ms cut-search frame and 120 s longest piece, in samples.
FRAME = 1_600
PIECE = 120 * SAMPLE_RATE


def _payload(samples: np.ndarray | None = None, *, duration_ms: int = 1_000) -> AudioPayload:
    if samples is None:
        samples = np.zeros(duration_ms * 16, dtype="<i2")
    samples = np.asarray(samples, dtype="<i2")
    return AudioPayload(
        pcm_s16le=samples.tobytes(),
        sample_rate=SAMPLE_RATE,
        sample_count=samples.shape[0],
        duration_ms=samples.shape[0] * 1_000 // SAMPLE_RATE,
        source_sample_rate=44_100,
        source_sample_count=samples.shape[0] * 3,
        source_channels=2,
        container="wav",
    )


def _prepared(*payloads: AudioPayload) -> list[PreparedItem[AudioPayload]]:
    return [
        PreparedItem(payload=payload, cost=payload.duration_ms, original_index=i) for i, payload in enumerate(payloads)
    ]


class _FakeTokenizer:
    """Chat prompt ``[5, <audio>, prompt bytes..., 6]``; decodes IDs as ``t<id>`` words."""

    eos_token_id = EOS_ID
    pad_token_id = PAD_ID

    def __init__(self) -> None:
        self.chats: list[list[dict[str, str]]] = []

    def apply_chat_template(self, chat: list[dict[str, str]], *, tokenize: bool, add_generation_prompt: bool) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        self.chats.append(chat)
        return f"USER: {chat[0]['content']}\n ASSISTANT:"

    def __call__(self, text: str) -> dict[str, list[int]]:
        body = text.removeprefix("USER: ").removesuffix("\n ASSISTANT:")
        ids = [5]
        while body:
            if body.startswith("<|audio|>"):
                ids.append(AUDIO_ID)
                body = body[len("<|audio|>") :]
            else:
                ids.append(10 + ord(body[0]) % 50)
                body = body[1:]
        return {"input_ids": [*ids, 6]}

    def batch_decode(self, generated: torch.Tensor, *, skip_special_tokens: bool) -> list[str]:
        assert skip_special_tokens is True
        return [" " + " ".join(f"t{int(t)}" for t in row if int(t) not in (EOS_ID, PAD_ID)) + " " for row in generated]


class _FakeModel:
    """Embeds token ``i`` as ``[i] * HIDDEN`` and audio frame ``f`` as ``[-1000 - f] * HIDDEN``."""

    dtype = torch.float32

    def __init__(self, outputs: list[list[int]] | None = None) -> None:
        self.embedding = torch.nn.Embedding(200, HIDDEN)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.arange(200, dtype=torch.float32).unsqueeze(1).repeat(1, HIDDEN))
        self.feature_shapes: list[tuple[int, ...]] = []
        self.language_model = SimpleNamespace(generate=MagicMock(side_effect=self._generate))
        self.outputs = outputs
        self.generate_calls: list[dict[str, Any]] = []

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.embedding

    def get_audio_features(self, features: torch.Tensor) -> torch.Tensor:
        self.feature_shapes.append(tuple(features.shape))
        frames = features.shape[1]
        # The projector emits three queries per 15-frame window.
        count = -(-frames // 15) * 3
        values = -1000.0 - torch.arange(count, dtype=torch.float32)
        return values.reshape(1, count, 1).repeat(1, 1, HIDDEN)

    def _generate(self, **kwargs: Any) -> torch.Tensor:
        self.generate_calls.append(kwargs)
        rows = kwargs["inputs_embeds"].shape[0]
        outputs = self.outputs or [[40 + row, EOS_ID] for row in range(rows)]
        return torch.tensor(outputs[:rows])


def _loaded(model: _FakeModel | None = None, **kwargs: Any) -> tuple[GraniteSpeechAdapter, _FakeModel, _FakeTokenizer]:
    adapter = GraniteSpeechAdapter(MODEL_ID, **kwargs)
    model = model or _FakeModel()
    tokenizer = _FakeTokenizer()
    adapter._model = model
    adapter._tokenizer = tokenizer
    adapter._device = "cpu"
    adapter._audio_token_id = AUDIO_ID
    adapter._window = torch.hann_window(400)
    adapter._mel_filters = mel_filterbank(257, 80, SAMPLE_RATE)
    adapter._preprocessor = AudioPreprocessor()
    return adapter, model, tokenizer


def _speech(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(round(seconds * SAMPLE_RATE)) * 3_000).astype("<i2")


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_capabilities_and_encode_contract() -> None:
    adapter = GraniteSpeechAdapter(MODEL_ID)

    assert adapter.capabilities.inputs == ["audio"]
    assert adapter.capabilities.outputs == ["json"]
    assert adapter.dims.dense is None
    with pytest.raises(NotImplementedError, match=r"Use extract\(\) instead"):
        adapter.encode([Item(text="hello")], ["dense"])


def test_constructor_rejects_invalid_loadtime_options() -> None:
    with pytest.raises(ValueError, match="max_padded_batch_ms must be positive"):
        GraniteSpeechAdapter(MODEL_ID, max_padded_batch_ms=0)
    with pytest.raises(ValueError, match="prompt must be a non-empty string"):
        GraniteSpeechAdapter(MODEL_ID, prompt="  ")


def test_extract_requires_loaded_prepared_audio() -> None:
    adapter = GraniteSpeechAdapter(MODEL_ID)
    with pytest.raises(RuntimeError, match="Model not loaded"):
        adapter.extract([Item()])

    adapter, _, _ = _loaded()
    with pytest.raises(ValueError, match="one Rust-prepared audio payload per item"):
        adapter.extract([Item()])
    with pytest.raises(TypeError, match="non-audio prepared item"):
        adapter.extract([Item()], prepared_items=[SimpleNamespace(payload=object())])


def test_extract_rejects_labels_and_schema() -> None:
    adapter, _, _ = _loaded()
    prepared = _prepared(_payload())
    with pytest.raises(ValueError, match="entity labels"):
        adapter.extract([Item()], labels=["person"], prepared_items=prepared)
    with pytest.raises(ValueError, match="output_schema"):
        adapter.extract([Item()], output_schema={"type": "object"}, prepared_items=prepared)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"beam_size": 4}, "unsupported Granite Speech options: beam_size"),
        ({"language": ""}, "language must be a non-empty string"),
        ({"language": 3}, "language must be a non-empty string"),
        ({"language": "it"}, "transcribes de, en, es, fr, ja, pt; got language 'it'"),
        ({"temperature": True}, "temperature must be a number"),
        ({"temperature": 1.5}, "temperature must be a number"),
        ({"timestamp_granularities": "word"}, "must be a list of strings"),
        ({"timestamp_granularities": ["segment"]}, "does not return timestamps"),
    ],
)
def test_extract_rejects_invalid_options(options: dict[str, Any], message: str) -> None:
    adapter, model, _ = _loaded()
    with pytest.raises(ValueError, match=message):
        adapter.extract([Item()], options=options, prepared_items=_prepared(_payload()))
    assert model.generate_calls == []


def test_extract_rejects_audio_not_at_sixteen_kilohertz() -> None:
    adapter, _, _ = _loaded()
    payload = dataclasses.replace(_payload(), sample_rate=8_000)
    with pytest.raises(ValueError, match="expects 16000 Hz"):
        adapter.extract([Item()], prepared_items=_prepared(payload))


# ---------------------------------------------------------------------------
# Prompting and decoding
# ---------------------------------------------------------------------------


def test_default_prompt_is_the_model_card_punctuated_asr_prompt() -> None:
    adapter, model, tokenizer = _loaded()

    out = adapter.extract([Item()], prepared_items=_prepared(_payload(_speech(1.0))))

    assert tokenizer.chats == [[{"role": "user", "content": f"<|audio|>{DEFAULT_PROMPT}"}]]
    assert DEFAULT_PROMPT == "transcribe the speech with proper punctuation and capitalization."
    assert out.data == [{"text": "t40", "language": None, "duration_ms": 1_000}]
    assert out.entities == [[]]
    assert out.batch_size == 1
    call = model.generate_calls[0]
    assert call["do_sample"] is False
    assert call["num_beams"] == 1
    assert (call["eos_token_id"], call["pad_token_id"]) == (EOS_ID, PAD_ID)
    # 64 tokens plus 12 per second of the longest row.
    assert call["max_new_tokens"] == 64 + 12


@pytest.mark.parametrize(
    ("instruction", "options", "content"),
    [
        ("translate the speech to French.", None, "<|audio|>translate the speech to French."),
        (
            None,
            {"language": "de"},
            f"<|audio|>The speech is in German. Transcribe it exactly; do not translate. {DEFAULT_PROMPT}",
        ),
        (
            None,
            {"language": "Japanese"},
            f"<|audio|>The speech is in Japanese. Transcribe it exactly; do not translate. {DEFAULT_PROMPT}",
        ),
        (
            "  ",
            {"language": "EN"},
            f"<|audio|>The speech is in English. Transcribe it exactly; do not translate. {DEFAULT_PROMPT}",
        ),
    ],
)
def test_instruction_replaces_the_prompt_and_language_prefixes_it(
    instruction: str | None, options: dict[str, Any] | None, content: str
) -> None:
    adapter, _, tokenizer = _loaded()

    out = adapter.extract(
        [Item()], instruction=instruction, options=options, prepared_items=_prepared(_payload(_speech(1.0)))
    )

    assert tokenizer.chats == [[{"role": "user", "content": content}]]
    expected_language = None if options is None else {"de": "de", "Japanese": "ja", "EN": "en"}[options["language"]]
    assert out.data[0]["language"] == expected_language


def test_loadtime_prompt_replaces_the_default() -> None:
    adapter, _, tokenizer = _loaded(prompt="can you transcribe the speech into a written format?")

    adapter.extract([Item()], prepared_items=_prepared(_payload(_speech(1.0))))

    assert tokenizer.chats[0][0]["content"] == "<|audio|>can you transcribe the speech into a written format?"


def test_positive_temperature_samples() -> None:
    adapter, model, _ = _loaded()

    adapter.extract([Item()], options={"temperature": 0.4}, prepared_items=_prepared(_payload(_speech(1.0))))

    assert model.generate_calls[0]["do_sample"] is True
    assert model.generate_calls[0]["temperature"] == 0.4


def test_audio_embeddings_replace_the_placeholder_and_rows_are_left_padded() -> None:
    adapter, model, _ = _loaded(prompt="ab")
    short, long = _speech(1.0, seed=1), _speech(3.0, seed=2)

    out = adapter.extract([Item(), Item()], prepared_items=_prepared(_payload(short), _payload(long)))

    # Each recording runs through the encoder alone, at its own length.
    assert model.feature_shapes == [(1, 150, 160), (1, 50, 160)]
    call = model.generate_calls[0]
    embeds, mask = call["inputs_embeds"], call["attention_mask"]
    prompt_tokens = [5.0, 10.0 + ord("a") % 50, 10.0 + ord("b") % 50, 6.0]
    long_audio = 30  # ceil(150 / 15) * 3
    short_audio = 12  # ceil(50 / 15) * 3
    assert embeds.shape == (2, 1 + long_audio + 3, HIDDEN)
    # Row 0 is the longest recording (decoded longest first), unpadded.
    assert embeds[0, :, 0].tolist() == [prompt_tokens[0], *(-1000.0 - np.arange(long_audio)), *prompt_tokens[1:]]
    assert mask[0].tolist() == [1] * embeds.shape[1]
    pad = long_audio - short_audio
    assert embeds[1, :pad].abs().sum().item() == 0
    assert mask[1].tolist() == [0] * pad + [1] * (embeds.shape[1] - pad)
    assert embeds[1, pad:, 0].tolist() == [prompt_tokens[0], *(-1000.0 - np.arange(short_audio)), *prompt_tokens[1:]]
    # Texts return to their requests' order.
    assert [row["text"] for row in out.data] == ["t41", "t40"]
    assert call["max_new_tokens"] == 64 + 36


def test_audio_too_short_for_features_transcribes_to_empty_text() -> None:
    adapter, model, _ = _loaded()

    out = adapter.extract([Item()], prepared_items=_prepared(_payload(_speech(256 / SAMPLE_RATE))))

    assert out.data == [{"text": "", "language": None, "duration_ms": 16}]
    assert model.generate_calls == []


def test_decoding_groups_rows_by_padded_length_budget() -> None:
    adapter, model, _ = _loaded(max_padded_batch_ms=4_000)
    payloads = [_payload(_speech(seconds, seed=i)) for i, seconds in enumerate([1.0, 3.0, 1.0, 2.0])]

    out = adapter.extract([Item() for _ in payloads], prepared_items=_prepared(*payloads))

    assert [call["inputs_embeds"].shape[0] for call in model.generate_calls] == [1, 2, 1]
    assert len(out.data) == 4


def test_length_chunks_never_returns_an_empty_chunk() -> None:
    assert _length_chunks([], 10) == []
    assert _length_chunks([20, 5, 5], 10) == [[0], [1, 2]]


# ---------------------------------------------------------------------------
# Long audio
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("length", [1, PIECE - 1, PIECE])
def test_pause_windows_keep_audio_of_at_most_two_minutes_whole(length: int) -> None:
    assert _pause_windows(np.ones(length, dtype=np.float32), SAMPLE_RATE) == [(0, length)]


def test_pause_windows_cut_at_the_quietest_frame_in_the_last_twenty_seconds() -> None:
    waveform = np.full(PIECE + 10 * SAMPLE_RATE, 0.5, dtype=np.float32)
    quiet = PIECE - 5 * SAMPLE_RATE
    waveform[quiet : quiet + FRAME] = 0.0
    # A quieter frame before the search window is ignored.
    waveform[SAMPLE_RATE : SAMPLE_RATE + FRAME] = 0.0

    windows = _pause_windows(waveform, SAMPLE_RATE)

    assert windows == [(0, quiet + FRAME // 2), (quiet + FRAME // 2, waveform.shape[0])]


@pytest.mark.parametrize("seconds", [120.0001, 239.9, 360.5, 720.0])
def test_pause_windows_cover_the_recording_with_contiguous_pieces_of_at_most_two_minutes(seconds: float) -> None:
    waveform = np.random.default_rng(0).standard_normal(round(seconds * SAMPLE_RATE)).astype(np.float32)

    windows = _pause_windows(waveform, SAMPLE_RATE)

    assert windows[0][0] == 0
    assert windows[-1][1] == waveform.shape[0]
    assert all(end == next_start for (_, end), (next_start, _) in pairwise(windows))
    assert all(0 < end - start <= PIECE for start, end in windows)


def test_long_request_is_decoded_as_pieces_and_joins_their_texts() -> None:
    adapter, _, _ = _loaded()
    seen: list[int] = []

    def fake_transcribe(pieces: list[np.ndarray], prompt_ids: list[int], temperature: float | None) -> list[str]:
        seen.extend(piece.shape[0] for piece in pieces)
        return [f"p{index}" if index != 1 else "" for index in range(len(pieces))]

    adapter._transcribe = fake_transcribe  # type: ignore[method-assign]
    long = _payload(_speech(250.0))
    short = _payload(_speech(2.0, seed=3))

    out = adapter.extract([Item(), Item()], prepared_items=_prepared(long, short))

    assert sum(seen[:3]) == long.sample_count
    assert all(length <= PIECE for length in seen)
    assert seen[3] == short.sample_count
    # The empty middle piece adds no extra space.
    assert [row["text"] for row in out.data] == ["p0 p2", "p3"]


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def test_mel_filterbank_matches_the_htk_reference() -> None:
    from transformers.audio_utils import mel_filter_bank

    reference = mel_filter_bank(
        num_frequency_bins=257,
        num_mel_filters=80,
        min_frequency=0.0,
        max_frequency=8_000.0,
        sampling_rate=SAMPLE_RATE,
        norm=None,
        mel_scale="htk",
    )

    np.testing.assert_allclose(mel_filterbank(257, 80, SAMPLE_RATE).numpy(), reference, atol=1e-5)


@pytest.mark.parametrize("samples", [257, 16_000, 16_001, 123_457])
def test_log_mel_features_match_a_numpy_reference(samples: int) -> None:
    from transformers.audio_utils import mel_filter_bank, spectrogram, window_function

    audio = np.random.default_rng(samples).standard_normal(samples).astype(np.float32) * 0.1
    mel = spectrogram(
        audio.astype(np.float64),
        window_function(400, "hann", periodic=True),
        frame_length=400,
        hop_length=160,
        fft_length=512,
        power=2.0,
        center=True,
        pad_mode="reflect",
        mel_filters=mel_filter_bank(257, 80, 0.0, 8_000.0, SAMPLE_RATE, norm=None, mel_scale="htk"),
    )
    logmel = np.log10(np.clip(mel.T, 1e-10, None))
    logmel = np.maximum(logmel, logmel.max() - 8.0) / 4 + 1
    if logmel.shape[0] % 2:
        logmel = logmel[:-1]
    reference = logmel.reshape(1, -1, 160)

    ours = log_mel_features(torch.from_numpy(audio), torch.hann_window(400), mel_filterbank(257, 80, SAMPLE_RATE))

    # Frame count is the processor's: (samples // 160 + 1) // 2.
    assert ours.shape == (1, (samples // 160 + 1) // 2, 160)
    np.testing.assert_allclose(ours.numpy(), reference, atol=2e-4)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _fake_transformers(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock, MagicMock]:
    model = MagicMock()
    model.config = SimpleNamespace(audio_token_id=100352)
    auto_tokenizer = MagicMock()
    model_class = MagicMock()
    model_class.from_pretrained.return_value = model
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=auto_tokenizer, GraniteSpeechForConditionalGeneration=model_class),
    )
    security = MagicMock()
    monkeypatch.setattr(granite_adapter, "install_transformers_security_fixes", security)
    return auto_tokenizer, model_class, security


@pytest.mark.parametrize(
    ("device", "precision", "dtype"),
    [("cpu", "bfloat16", torch.float32), ("cuda:0", "bfloat16", torch.bfloat16), ("cuda", "float16", torch.float16)],
)
def test_load_pins_revision_and_lets_transformers_pick_attention(
    monkeypatch: pytest.MonkeyPatch, device: str, precision: str, dtype: torch.dtype
) -> None:
    auto_tokenizer, model_class, security = _fake_transformers(monkeypatch)
    monkeypatch.setattr(torch, "hann_window", lambda length, device=None: torch.ones(length))
    adapter = GraniteSpeechAdapter(MODEL_ID, compute_precision=precision, revision="abc123")
    monkeypatch.setattr(granite_adapter, "mel_filterbank", lambda *args: MagicMock())

    adapter.load(device)

    security.assert_called_once_with()
    auto_tokenizer.from_pretrained.assert_called_once_with(MODEL_ID, revision="abc123")
    # No attn_implementation: forcing SDPA on the whole model fails, since the Q-Former has none.
    model_class.from_pretrained.assert_called_once_with(
        MODEL_ID, dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True, revision="abc123"
    )
    model = model_class.from_pretrained.return_value
    model.to.assert_called_once_with(device)
    model.eval.assert_called_once_with()
    assert adapter._audio_token_id == 100352
    preprocessor = adapter.get_preprocessor()
    assert isinstance(preprocessor, AudioPreprocessor)
    assert preprocessor._runs_alone_above_samples == PIECE

    adapter.unload()
    assert adapter._model is None
    assert adapter._tokenizer is None


def test_load_forwards_an_explicit_attention_implementation(monkeypatch: pytest.MonkeyPatch) -> None:
    _, model_class, _ = _fake_transformers(monkeypatch)
    adapter = GraniteSpeechAdapter(MODEL_ID, attn_implementation="eager")

    adapter.load("cpu")

    assert model_class.from_pretrained.call_args.kwargs["attn_implementation"] == "eager"


# ---------------------------------------------------------------------------
# Catalog and bundle
# ---------------------------------------------------------------------------


def test_model_config_is_pinned_audio_extract_and_routed_by_default_bundle() -> None:
    config = load_model_configs(MODELS)[MODEL_ID]

    assert config.hf_id == MODEL_ID
    assert config.hf_revision == "c684620bb54fc89d14ba27256b23bbc9e09ba764"
    assert config.inputs.audio is True
    assert config.inputs.text is False
    assert config.tasks.extract is not None
    assert config.tasks.encode is None
    assert config.tasks.score is None
    profile = config.resolve_profile("default")
    assert profile.adapter_path == "sie_server.adapters.granite_speech.adapter:GraniteSpeechAdapter"
    assert profile.compute_precision == "bfloat16"
    # One 12-minute request (SIE's per-request audio cap) per extract call.
    assert profile.max_batch_tokens == 12 * 60 * 1_000
    adapter_class = resolve_adapter_class(config, MODELS)
    assert adapter_class is GraniteSpeechAdapter
    reject_unknown_loadtime_options(adapter_class, profile.loadtime, model_name=MODEL_ID)
    assert MODEL_ID in match_bundle_models(BUNDLES / "default.yaml", MODELS)
    assert MODEL_ID not in match_bundle_models(BUNDLES / "transformers5.yaml", MODELS)
