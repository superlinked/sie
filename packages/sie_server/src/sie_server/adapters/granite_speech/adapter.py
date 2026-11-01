"""Granite Speech speech-to-text adapter implemented on the native extract primitive.

``ibm-granite/granite-speech-5.0-2b`` is a speech-language model: a conformer
encoder and a windowed Q-Former projector turn 16 kHz audio into ten LLM input
embeddings per second, and a Granite 4.0 1B language model writes the answer to
a text instruction about that audio. The instruction selects the task, so the
same weights transcribe, transcribe with keyword biasing, or translate.

The request ``instruction`` replaces the default prompt, as the OpenAI
transcription ``prompt`` field does on IBM's reference endpoints. Without one,
the model transcribes with punctuation and capitalization, the model card's
preferred ASR prompt. A ``language`` option among the six the model transcribes
prefixes the model card's language-ID sentence, which keeps the model from
translating. ``temperature`` above zero samples; otherwise decoding is greedy,
as in the model card.

Features are the log-mel spectrogram of ``GraniteSpeechFeatureExtractor``,
computed here with ``torch.stft`` instead of ``torchaudio``, which no SIE bundle
ships. The filterbank, window, padding and normalization reproduce
``torchaudio.transforms.MelSpectrogram`` with the extractor's settings.

Each recording runs through the encoder and projector on its own, at its true
length, because padding a batch of recordings changes what the conformer sees.
Only language-model decoding is batched, with left-padded prompts.

Audio longer than two minutes, the longest segment the model card recommends,
is decoded in pieces of at most 120 s, each cut at the quietest 100 ms in the
last 20 s before its 120 s mark, with no overlap. A request's text joins its
pieces' texts with spaces. The model returns no timestamps.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._transformers_compat import install_transformers_security_fixes
from sie_server.adapters._types import ComputePrecision
from sie_server.core.inference_output import EncodeOutput, ExtractOutput
from sie_server.core.prepared import AudioPayload
from sie_server.core.preprocessor.audio import AudioPreprocessor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sie_server.types.inputs import Item

logger = logging.getLogger(__name__)

_ERR_ENCODE_NOT_SUPPORTED = "GraniteSpeechAdapter does not support encode(). Use extract() instead."
_RUNTIME_OPTIONS = frozenset({"language", "temperature", "timestamp_granularities"})

DEFAULT_PROMPT = "transcribe the speech with proper punctuation and capitalization."
_AUDIO_TOKEN = "<|audio|>"  # noqa: S105 - placeholder token, not a secret
# Languages the model transcribes, keyed by ISO 639-1 code and English name.
_LANGUAGES = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "ja": "Japanese",
}
_LANGUAGE_PROMPT = "The speech is in {language}. Transcribe it exactly; do not translate. {prompt}"

# GraniteSpeechFeatureExtractor settings (preprocessor_config.json).
_SAMPLE_RATE = 16_000
_N_FFT = 512
_WIN_LENGTH = 400
_HOP_LENGTH = 160
_N_MELS = 80
# Reflect padding of the centered STFT needs more than half a window; shorter
# audio (16 ms) has no features and transcribes to empty text.
_MIN_SAMPLES = _N_FFT // 2 + 1

# Audio is decoded in pieces of at most _PIECE_MAX_S seconds, each cut in the
# middle of the quietest _CUT_FRAME_S frame in the last _CUT_SEARCH_S before
# its _PIECE_MAX_S mark.
_PIECE_MAX_S = 120.0
_CUT_SEARCH_S = 20.0
_CUT_FRAME_S = 0.1

# New-token budget per piece: generous for fast speech in any supported
# language, so greedy decoding normally stops at end-of-text first.
_NEW_TOKENS_BASE = 64
_NEW_TOKENS_PER_SECOND = 12.0


class GraniteSpeechAdapter(BaseAdapter):
    """Batched transcription for Granite Speech 5.0."""

    spec: ClassVar[AdapterSpec] = AdapterSpec(
        inputs=("audio",),
        outputs=("json",),
        unload_fields=("_model", "_tokenizer", "_preprocessor", "_mel_filters", "_window"),
        default_preprocessor="audio",
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        compute_precision: ComputePrecision = "bfloat16",
        attn_implementation: str | None = None,
        revision: str | None = None,
        prompt: str = DEFAULT_PROMPT,
        max_padded_batch_ms: int = 480_000,
        **kwargs: Any,
    ) -> None:
        del kwargs
        if max_padded_batch_ms <= 0:
            msg = "max_padded_batch_ms must be positive"
            raise ValueError(msg)
        if not isinstance(prompt, str) or not prompt.strip():
            msg = "prompt must be a non-empty string"
            raise ValueError(msg)
        self._model_name_or_path = str(model_name_or_path)
        self._compute_precision = compute_precision
        self._attn_implementation = attn_implementation
        self._revision = revision
        self._prompt = prompt.strip()
        self._max_padded_batch_ms = max_padded_batch_ms
        self._model: Any = None
        self._tokenizer: Any = None
        self._preprocessor: AudioPreprocessor | None = None
        self._mel_filters: torch.Tensor | None = None
        self._window: torch.Tensor | None = None
        self._audio_token_id: int | None = None
        self._device: str | None = None

    def load(self, device: str) -> None:
        from transformers import AutoTokenizer, GraniteSpeechForConditionalGeneration

        self._device = device
        dtype = self._resolve_dtype()
        shared_kwargs: dict[str, Any] = {}
        if self._revision is not None:
            shared_kwargs["revision"] = self._revision
        model_kwargs: dict[str, Any] = {}
        # Transformers picks per submodule by default: SDPA for the language
        # model, eager for the Q-Former, which has no SDPA path. Forcing one
        # implementation on the whole model fails to load.
        if self._attn_implementation is not None:
            model_kwargs["attn_implementation"] = self._attn_implementation

        logger.info(
            "Loading Granite Speech model %s on device=%s with dtype=%s, attn=%s",
            self._model_name_or_path,
            device,
            dtype,
            self._attn_implementation or "auto",
        )
        install_transformers_security_fixes()
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name_or_path, **shared_kwargs)
        self._model = GraniteSpeechForConditionalGeneration.from_pretrained(
            self._model_name_or_path,
            dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            **model_kwargs,
            **shared_kwargs,
        )
        self._model.to(device)
        self._model.eval()
        self._audio_token_id = int(self._model.config.audio_token_id)
        self._window = torch.hann_window(_WIN_LENGTH, device=device)
        self._mel_filters = mel_filterbank(_N_FFT // 2 + 1, _N_MELS, _SAMPLE_RATE).to(device)
        # A recording past one piece is split and decoded alone, so short clips
        # never wait behind it.
        self._preprocessor = AudioPreprocessor(runs_alone_above_samples=round(_PIECE_MAX_S * _SAMPLE_RATE))

    def _resolve_dtype(self) -> torch.dtype:
        if not self._device or not self._device.startswith("cuda"):
            return torch.float32
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(self._compute_precision, torch.bfloat16)

    def get_preprocessor(self) -> AudioPreprocessor | None:
        return self._preprocessor

    def encode(
        self,
        items: list[Item],
        output_types: list[str],
        *,
        instruction: str | None = None,
        is_query: bool = False,
        prepared_items: list[Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> EncodeOutput:
        raise NotImplementedError(_ERR_ENCODE_NOT_SUPPORTED)

    def extract(
        self,
        items: list[Item],
        *,
        labels: list[str] | None = None,
        output_schema: dict[str, Any] | None = None,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
        prepared_items: list[Any] | None = None,
    ) -> ExtractOutput:
        self._check_loaded()
        if self._tokenizer is None:
            msg = "GraniteSpeechAdapter is not loaded"
            raise RuntimeError(msg)
        if labels:
            msg = "Granite Speech transcription does not accept entity labels"
            raise ValueError(msg)
        if output_schema:
            msg = "Granite Speech transcription does not accept output_schema"
            raise ValueError(msg)

        payloads = _audio_payloads(items, prepared_items)
        language, temperature = _parse_options(options)
        prompt = instruction.strip() if instruction and instruction.strip() else self._prompt
        if language is not None:
            prompt = _LANGUAGE_PROMPT.format(language=_LANGUAGES[language], prompt=prompt)
        prompt_ids = self._prompt_ids(prompt)

        pieces: list[np.ndarray] = []
        owners: list[int] = []
        for index, payload in enumerate(payloads):
            if payload.sample_rate != _SAMPLE_RATE:
                msg = f"Granite Speech expects {_SAMPLE_RATE} Hz audio, got {payload.sample_rate} Hz"
                raise ValueError(msg)
            waveform = _waveform(payload)
            for start, end in _pause_windows(waveform, _SAMPLE_RATE):
                pieces.append(waveform[start:end])
                owners.append(index)

        texts = self._transcribe(pieces, prompt_ids, temperature)
        joined: list[list[str]] = [[] for _ in payloads]
        for owner, text in zip(owners, texts, strict=True):
            if text:
                joined[owner].append(text)

        data = [
            {"text": " ".join(parts), "language": language, "duration_ms": payload.duration_ms}
            for parts, payload in zip(joined, payloads, strict=True)
        ]
        return ExtractOutput(
            entities=[[] for _ in data],
            data=data,
            batch_size=len(data),
        )

    def _prompt_ids(self, prompt: str) -> list[int]:
        """Token IDs of the chat prompt, with one audio placeholder, as the processor builds them."""
        chat = [{"role": "user", "content": f"{_AUDIO_TOKEN}{prompt}"}]
        text = self._tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        ids = [int(token) for token in self._tokenizer(text)["input_ids"]]
        if ids.count(self._audio_token_id) != 1:
            msg = "Granite Speech prompt must contain exactly one audio placeholder"
            raise ValueError(msg)
        return ids

    @torch.inference_mode()
    def _transcribe(
        self,
        pieces: list[np.ndarray],
        prompt_ids: list[int],
        temperature: float | None,
    ) -> list[str]:
        texts = [""] * len(pieces)
        decodable = [index for index, piece in enumerate(pieces) if piece.shape[0] >= _MIN_SAMPLES]
        max_padded = round(self._max_padded_batch_ms * _SAMPLE_RATE / 1000)
        lengths = [pieces[index].shape[0] for index in decodable]
        for chunk in _length_chunks(lengths, max_padded):
            rows = [decodable[position] for position in chunk]
            outputs = self._generate([pieces[row] for row in rows], prompt_ids, temperature)
            for row, text in zip(rows, outputs, strict=True):
                texts[row] = text
        return texts

    def _generate(
        self,
        waveforms: list[np.ndarray],
        prompt_ids: list[int],
        temperature: float | None,
    ) -> list[str]:
        model = self._model
        device = self._device
        embed_tokens = model.get_input_embeddings()
        split = prompt_ids.index(self._audio_token_id)
        prefix = torch.tensor(prompt_ids[:split], device=device, dtype=torch.long)
        suffix = torch.tensor(prompt_ids[split + 1 :], device=device, dtype=torch.long)
        prefix_embeds = embed_tokens(prefix)
        suffix_embeds = embed_tokens(suffix)

        rows: list[torch.Tensor] = []
        for waveform in waveforms:
            features = self._features(waveform).to(model.dtype)
            audio_embeds = model.get_audio_features(features)[0]
            rows.append(torch.cat([prefix_embeds, audio_embeds.to(prefix_embeds.dtype), suffix_embeds], dim=0))

        longest = max(row.shape[0] for row in rows)
        hidden = rows[0].shape[1]
        inputs_embeds = prefix_embeds.new_zeros((len(rows), longest, hidden))
        attention_mask = torch.zeros((len(rows), longest), dtype=torch.long, device=device)
        for position, row in enumerate(rows):
            inputs_embeds[position, longest - row.shape[0] :] = row
            attention_mask[position, longest - row.shape[0] :] = 1

        max_seconds = max(waveform.shape[0] for waveform in waveforms) / _SAMPLE_RATE
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": _NEW_TOKENS_BASE + math.ceil(_NEW_TOKENS_PER_SECOND * max_seconds),
            "num_beams": 1,
            "eos_token_id": self._tokenizer.eos_token_id,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if temperature:
            generation_kwargs.update(do_sample=True, temperature=temperature)
        else:
            generation_kwargs["do_sample"] = False
        generated = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generation_kwargs,
        )
        # With only inputs_embeds, generate() returns just the new tokens.
        return [text.strip() for text in self._tokenizer.batch_decode(generated, skip_special_tokens=True)]

    def _features(self, waveform: np.ndarray) -> torch.Tensor:
        """``GraniteSpeechFeatureExtractor`` features of one recording: ``(1, frames, 160)``."""
        if self._mel_filters is None or self._window is None:
            msg = "GraniteSpeechAdapter is not loaded"
            raise RuntimeError(msg)
        audio = torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32)).to(self._device)
        return log_mel_features(audio, self._window, self._mel_filters)


def mel_filterbank(n_freqs: int, n_mels: int, sample_rate: int) -> torch.Tensor:
    """``torchaudio.functional.melscale_fbanks`` with HTK mels, no norm, 0 Hz to Nyquist."""
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)
    m_min = 2595.0 * math.log10(1.0)
    m_max = 2595.0 * math.log10(1.0 + (sample_rate / 2.0) / 700.0)
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    f_pts = 700.0 * (10.0 ** (m_pts / 2595.0) - 1.0)
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]
    up_slopes = slopes[:, 2:] / f_diff[1:]
    return torch.max(torch.zeros(1), torch.min(down_slopes, up_slopes))


def log_mel_features(audio: torch.Tensor, window: torch.Tensor, mel_filters: torch.Tensor) -> torch.Tensor:
    """Granite Speech log-mel features of one float32 waveform, as ``(1, frames // 2, 2 * n_mels)``."""
    spec = torch.stft(
        audio,
        n_fft=_N_FFT,
        hop_length=_HOP_LENGTH,
        win_length=_WIN_LENGTH,
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    power = spec.abs().pow(2.0)
    mel = torch.matmul(power.transpose(-1, -2), mel_filters).transpose(-1, -2)
    logmel = mel.transpose(-1, -2).clip(min=1e-10).log10()
    logmel = torch.maximum(logmel, logmel.amax() - 8.0).div(4).add(1)
    if logmel.shape[0] % 2 == 1:
        logmel = logmel[:-1]
    return logmel.reshape(1, -1, 2 * logmel.shape[-1])


def _audio_payloads(items: list[Item], prepared_items: list[Any] | None) -> list[AudioPayload]:
    if prepared_items is None or len(prepared_items) != len(items):
        msg = "GraniteSpeechAdapter requires one Rust-prepared audio payload per item"
        raise ValueError(msg)
    payloads = []
    for prepared in prepared_items:
        payload = getattr(prepared, "payload", None)
        if not isinstance(payload, AudioPayload):
            msg = "GraniteSpeechAdapter received a non-audio prepared item"
            raise TypeError(msg)
        payloads.append(payload)
    return payloads


def _parse_options(options: dict[str, Any] | None) -> tuple[str | None, float | None]:
    options = options or {}
    unknown = set(options) - _RUNTIME_OPTIONS
    if unknown:
        msg = f"unsupported Granite Speech options: {', '.join(sorted(unknown))}"
        raise ValueError(msg)

    language = options.get("language")
    normalized_language: str | None = None
    if language is not None:
        if not isinstance(language, str) or not language.strip():
            msg = "language must be a non-empty string or null"
            raise ValueError(msg)
        key = language.strip().lower()
        by_name = {name.lower(): code for code, name in _LANGUAGES.items()}
        normalized_language = key if key in _LANGUAGES else by_name.get(key)
        if normalized_language is None:
            msg = f"Granite Speech transcribes {', '.join(sorted(_LANGUAGES))}; got language {language!r}"
            raise ValueError(msg)

    raw_temperature = options.get("temperature")
    temperature = None
    if raw_temperature is not None:
        if isinstance(raw_temperature, bool) or not isinstance(raw_temperature, (int, float)):
            msg = "temperature must be a number between 0 and 1"
            raise ValueError(msg)
        temperature = float(raw_temperature)
        if not 0 <= temperature <= 1:
            msg = "temperature must be a number between 0 and 1"
            raise ValueError(msg)

    granularities = options.get("timestamp_granularities", [])
    if not isinstance(granularities, list) or not all(isinstance(value, str) for value in granularities):
        msg = "timestamp_granularities must be a list of strings"
        raise ValueError(msg)
    if granularities:
        msg = "Granite Speech does not return timestamps"
        raise ValueError(msg)
    return normalized_language, temperature


def _waveform(payload: AudioPayload) -> np.ndarray:
    return np.frombuffer(payload.pcm_s16le, dtype="<i2").astype(np.float32) / 32_768.0


def _pause_windows(waveform: np.ndarray, sample_rate: int) -> list[tuple[int, int]]:
    """Split a recording into contiguous, non-overlapping ``(start, end)`` sample spans.

    While more than ``_PIECE_MAX_S`` remains, the next span ends in the middle
    of the quietest ``_CUT_FRAME_S`` frame (by RMS, the first on ties) among the
    frames in the last ``_CUT_SEARCH_S`` before the ``_PIECE_MAX_S`` mark. A
    recording of at most ``_PIECE_MAX_S`` is one span.
    """
    total = waveform.shape[0]
    piece = round(_PIECE_MAX_S * sample_rate)
    if total <= piece:
        return [(0, total)]
    hop = round(_CUT_FRAME_S * sample_rate)
    search = round(_CUT_SEARCH_S * sample_rate)
    frames = total // hop
    scaled = waveform[: frames * hop].reshape(frames, hop) * np.float32(32_768.0)
    rms = np.sqrt(np.mean(np.square(scaled), axis=1) + 1e-9)
    windows: list[tuple[int, int]] = []
    start = 0
    while total - start > piece:
        low = (start + piece - search) // hop
        high = (start + piece) // hop
        cut = (low + int(np.argmin(rms[low:high]))) * hop + hop // 2
        windows.append((start, cut))
        start = cut
    windows.append((start, total))
    return windows


def _length_chunks(lengths: Sequence[int], max_padded: int) -> list[list[int]]:
    """Group indices, longest first, so ``rows * longest row`` stays within ``max_padded``.

    A row longer than the budget is decoded on its own.
    """
    chunks: list[list[int]] = []
    current: list[int] = []
    for index in sorted(range(len(lengths)), key=lambda position: lengths[position], reverse=True):
        if current and (len(current) + 1) * lengths[current[0]] > max_padded:
            chunks.append(current)
            current = []
        current.append(index)
    if current:
        chunks.append(current)
    return chunks
