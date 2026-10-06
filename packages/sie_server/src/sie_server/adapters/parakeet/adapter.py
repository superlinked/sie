"""Parakeet-TDT speech-to-text adapter implemented on the native extract primitive.

``nvidia/parakeet-tdt-0.6b-v3`` transcribes 25 European languages with
punctuation and capitalization. It identifies the spoken language itself and
cannot be steered to one, so a ``language`` option inside its set is accepted
and has no effect, and the response reports ``language: null``.

Decoding is the batched greedy TDT loop in :mod:`.decoding` rather than
``generate()``, whose TDT path has no per-frame symbol limit and can repeat a
token until the output buffer is full.

A request's audio is decoded in one pass, however long. Rows are grouped by
length so that a forward pass never holds more than ``max_padded_batch_ms`` of
padded audio: encoder self-attention grows with the square of a row's length,
so with that budget no batch needs more memory than a single recording of the
budget's length.

The model occasionally returns nothing for a short, tightly cut clip that
contains speech. Each empty row of at most ``_EMPTY_RETRY_MAX_S`` (30 s) is
decoded again on its own, first with 0.25 s of silence appended and then, if
still empty, with 0.25 s on both sides. Padding is never added to the batched
pass, where it lowers accuracy. A longer empty row is returned as it is: every
empty row the padding recovered in testing was under 12 s, and a long
recording that decodes to nothing (silence, music) would otherwise run
through the encoder three times.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ComputePrecision
from sie_server.adapters.parakeet.decoding import TdtHypothesis, greedy_tdt_decode
from sie_server.core.inference_output import EncodeOutput, ExtractOutput
from sie_server.core.prepared import AudioPayload
from sie_server.core.preprocessor.audio import AudioPreprocessor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sie_server.types.inputs import Item

logger = logging.getLogger(__name__)

# ISO 639-1 codes of the languages the checkpoint was trained to transcribe.
SUPPORTED_LANGUAGES = frozenset(
    {
        "bg",
        "cs",
        "da",
        "de",
        "el",
        "en",
        "es",
        "et",
        "fi",
        "fr",
        "hr",
        "hu",
        "it",
        "lt",
        "lv",
        "mt",
        "nl",
        "pl",
        "pt",
        "ro",
        "ru",
        "sk",
        "sl",
        "sv",
        "uk",
    }
)
_ERR_ENCODE_NOT_SUPPORTED = "ParakeetTDTAdapter does not support encode(). Use extract() instead."
_RUNTIME_OPTIONS = frozenset({"language", "temperature", "timestamp_granularities"})
_TIMESTAMP_GRANULARITIES = frozenset({"segment", "word"})
# Seconds of silence (leading, trailing) for the re-decodes of an empty row, in order.
_EMPTY_RETRY_PADDING_S = ((0.0, 0.25), (0.25, 0.25))
# Longest empty row, in seconds, that is re-decoded with padding.
_EMPTY_RETRY_MAX_S = 30.0
# A segment ends at a word ending in sentence-final punctuation (. ! ? or an
# ellipsis, optionally followed by closing quotes or brackets), or once it spans
# this many seconds.
_SENTENCE_END_RE = re.compile(r"[.!?\u2026][\"'\u00bb\u201d\u2019)\]]*$")
_MAX_SEGMENT_S = 30.0
_DEFAULT_MAX_PADDED_BATCH_MS = 12 * 60 * 1_000


@dataclass(slots=True)
class _Transcript:
    text: str = ""
    hypothesis: TdtHypothesis = field(default_factory=TdtHypothesis)
    # Seconds of silence prepended before decoding; subtracted from timestamps.
    offset_s: float = 0.0


class ParakeetTDTAdapter(BaseAdapter):
    """Batched transcription for Parakeet-TDT checkpoints (``ParakeetForTDT``)."""

    spec: ClassVar[AdapterSpec] = AdapterSpec(
        inputs=("audio",),
        outputs=("json",),
        unload_fields=("_model", "_processor", "_preprocessor"),
        default_preprocessor="audio",
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        compute_precision: ComputePrecision = "bfloat16",
        revision: str | None = None,
        max_padded_batch_ms: int = _DEFAULT_MAX_PADDED_BATCH_MS,
        **kwargs: Any,
    ) -> None:
        del kwargs
        if (
            isinstance(max_padded_batch_ms, bool)
            or not isinstance(max_padded_batch_ms, int)
            or max_padded_batch_ms <= 0
        ):
            msg = "max_padded_batch_ms must be a positive integer"
            raise ValueError(msg)
        self._model_name_or_path = str(model_name_or_path)
        self._compute_precision = compute_precision
        self._revision = revision
        self._max_padded_batch_ms = max_padded_batch_ms
        self._model: Any = None
        self._processor: Any = None
        self._preprocessor: AudioPreprocessor | None = None
        self._device: str | None = None
        self._dtype = torch.float32
        self._sample_rate = 16_000
        self._min_samples = 320
        self._blank_id = 0
        self._vocab_size = 0
        self._durations: tuple[int, ...] = ()

    def load(self, device: str) -> None:
        from transformers import (
            AutoModelForTDT,  # ty: ignore[unresolved-import]
            AutoProcessor,
        )

        self._device = device
        self._dtype = self._resolve_dtype()
        revision_kwargs: dict[str, Any] = {}
        if self._revision is not None:
            revision_kwargs["revision"] = self._revision

        logger.info(
            "Loading Parakeet-TDT model %s on device=%s with dtype=%s",
            self._model_name_or_path,
            device,
            self._dtype,
        )
        processor = AutoProcessor.from_pretrained(self._model_name_or_path, **revision_kwargs)
        model = AutoModelForTDT.from_pretrained(self._model_name_or_path, dtype=self._dtype, **revision_kwargs)
        model.to(device)
        model.eval()

        config = model.config
        feature_extractor = processor.feature_extractor
        self._blank_id = int(config.blank_token_id)
        self._vocab_size = int(config.vocab_size)
        self._durations = tuple(int(value) for value in config.durations)
        self._sample_rate = int(feature_extractor.sampling_rate)
        # Feature normalization needs at least two spectrogram frames.
        self._min_samples = 2 * int(feature_extractor.hop_length)
        if getattr(processor, "decoder_type", None) is None:
            # The checkpoint's processor config predates ``decoder_type``. Left
            # unset, the processor infers it from the full vocabulary for every
            # token it decodes with timestamps: seconds of CPU time per long
            # recording, more than the transcription itself.
            processor.decoder_type = "tdt"
        self._processor = processor
        self._model = model
        self._preprocessor = AudioPreprocessor()

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
        if self._processor is None:
            msg = "ParakeetTDTAdapter is not loaded"
            raise RuntimeError(msg)
        if labels:
            msg = "Parakeet transcription does not accept entity labels"
            raise ValueError(msg)
        if output_schema:
            msg = "Parakeet transcription does not accept output_schema"
            raise ValueError(msg)
        granularities = _parse_request(options, instruction)
        payloads = _audio_payloads(items, prepared_items)
        for payload in payloads:
            if payload.sample_rate != self._sample_rate:
                msg = f"Parakeet transcription requires {self._sample_rate} Hz audio, got {payload.sample_rate} Hz"
                raise ValueError(msg)

        waveforms = [_waveform(payload) for payload in payloads]
        transcripts = self._transcribe(waveforms)
        data = [
            self._transcript_data(transcript, payload, granularities)
            for transcript, payload in zip(transcripts, payloads, strict=True)
        ]
        return ExtractOutput(
            entities=[[] for _ in data],
            data=data,
            batch_size=len(data),
        )

    def _transcribe(self, waveforms: list[np.ndarray]) -> list[_Transcript]:
        transcripts = [_Transcript() for _ in waveforms]
        # A clip too short to normalize holds no speech; it is returned empty.
        decodable = [index for index, waveform in enumerate(waveforms) if waveform.shape[0] >= self._min_samples]
        budget = self._max_padded_batch_ms * self._sample_rate // 1_000
        for chunk in _length_chunks([waveforms[index].shape[0] for index in decodable], budget):
            rows = [decodable[position] for position in chunk]
            for row, transcript in zip(rows, self._decode([waveforms[row] for row in rows]), strict=True):
                transcripts[row] = transcript

        retry_max_samples = round(_EMPTY_RETRY_MAX_S * self._sample_rate)
        retried = 0
        for row in decodable:
            if transcripts[row].text or waveforms[row].shape[0] > retry_max_samples:
                continue
            retried += 1
            for leading_s, trailing_s in _EMPTY_RETRY_PADDING_S:
                padded = _pad(waveforms[row], self._sample_rate, leading_s, trailing_s)
                (transcript,) = self._decode([padded], offset_s=leading_s)
                if transcript.text:
                    transcripts[row] = transcript
                    break
        if retried:
            logger.debug("Re-decoded %d empty Parakeet transcript(s) with silence padding", retried)
        return transcripts

    def _decode(self, waveforms: list[np.ndarray], *, offset_s: float = 0.0) -> list[_Transcript]:
        with torch.inference_mode():
            inputs = self._processor(audio=waveforms, sampling_rate=self._sample_rate, return_tensors="pt")
            features = inputs["input_features"].to(self._device, dtype=self._dtype)
            attention_mask = inputs["attention_mask"].to(self._device)
            encoded = self._model.get_audio_features(
                input_features=features,
                attention_mask=attention_mask,
                output_attention_mask=True,
            )
            encoder_output = encoded.pooler_output
            if encoded.attention_mask is not None:
                lengths = encoded.attention_mask.sum(dim=-1)
            else:
                lengths = torch.full((encoder_output.shape[0],), encoder_output.shape[1], dtype=torch.long)
            hypotheses = greedy_tdt_decode(
                self._model.decoder,
                self._model.joint,
                encoder_output,
                lengths,
                blank_id=self._blank_id,
                vocab_size=self._vocab_size,
                durations=self._durations,
            )
        if len(hypotheses) != len(waveforms):
            msg = "Parakeet decoding returned a misaligned batch"
            raise RuntimeError(msg)
        with self._tokenizer_guard():
            # The processor keeps repeated tokens; the bare tokenizer would merge
            # them as if the output were CTC.
            texts = self._processor.batch_decode(
                [hypothesis.token_ids for hypothesis in hypotheses],
                skip_special_tokens=True,
                group_tokens=False,
            )
        return [
            _Transcript(text=str(text).strip(), hypothesis=hypothesis, offset_s=offset_s)
            for text, hypothesis in zip(texts, hypotheses, strict=True)
        ]

    def _transcript_data(
        self,
        transcript: _Transcript,
        payload: AudioPayload,
        granularities: frozenset[str],
    ) -> dict[str, Any]:
        data: dict[str, Any] = {
            "text": transcript.text,
            "language": None,
            "duration_ms": payload.duration_ms,
        }
        if not granularities:
            return data
        words = _words_from_tokens(self._timed_tokens(transcript, payload.duration_s))
        if "word" in granularities:
            data["words"] = words
        if "segment" in granularities:
            data["segments"] = _segments_from_words(words)
        return data

    def _timed_tokens(self, transcript: _Transcript, duration_s: float) -> list[dict[str, Any]]:
        """Token start/end seconds from each decode step's frame and duration."""
        hypothesis = transcript.hypothesis
        if not transcript.text or not hypothesis.step_ids:
            return []
        with self._tokenizer_guard():
            _, offsets = self._processor.decode(
                torch.tensor([hypothesis.step_ids], dtype=torch.long),
                durations=torch.tensor([hypothesis.step_durations], dtype=torch.long),
                skip_special_tokens=True,
                group_tokens=False,
            )
        tokens = []
        for offset in offsets[0]:
            start = min(max(float(offset["start"]) - transcript.offset_s, 0.0), duration_s)
            end = min(max(float(offset["end"]) - transcript.offset_s, start), duration_s)
            tokens.append({"token": str(offset["token"]), "start": start, "end": end})
        return tokens


def _audio_payloads(items: list[Item], prepared_items: list[Any] | None) -> list[AudioPayload]:
    if prepared_items is None or len(prepared_items) != len(items):
        msg = "ParakeetTDTAdapter requires one Rust-prepared audio payload per item"
        raise ValueError(msg)
    payloads = []
    for prepared in prepared_items:
        payload = getattr(prepared, "payload", None)
        if not isinstance(payload, AudioPayload):
            msg = "ParakeetTDTAdapter received a non-audio prepared item"
            raise TypeError(msg)
        payloads.append(payload)
    return payloads


def _parse_request(options: dict[str, Any] | None, instruction: str | None) -> frozenset[str]:
    """Validate the request options and return the timestamp granularities."""
    if instruction is not None and instruction.strip():
        msg = "this model does not accept a prompt (instruction); omit it"
        raise ValueError(msg)
    options = options or {}
    unknown = set(options) - _RUNTIME_OPTIONS
    if unknown:
        msg = f"unsupported Parakeet options: {', '.join(sorted(unknown))}"
        raise ValueError(msg)

    language = options.get("language")
    if language is not None and (not isinstance(language, str) or language.strip().lower() not in SUPPORTED_LANGUAGES):
        msg = (
            f"language {language!r} is not supported: this model transcribes {len(SUPPORTED_LANGUAGES)} languages "
            f"({', '.join(sorted(SUPPORTED_LANGUAGES))}; ISO 639-1 codes) and identifies the spoken one itself, "
            "so language may be omitted"
        )
        raise ValueError(msg)

    temperature = options.get("temperature")
    if temperature is not None and (
        isinstance(temperature, bool) or not isinstance(temperature, int | float) or temperature != 0
    ):
        msg = "this model decodes greedily; temperature must be 0 or omitted"
        raise ValueError(msg)

    raw_granularities = options.get("timestamp_granularities") or []
    if not isinstance(raw_granularities, list | tuple) or not all(
        isinstance(value, str) for value in raw_granularities
    ):
        msg = "timestamp_granularities must be a list of strings"
        raise ValueError(msg)
    granularities = frozenset(raw_granularities)
    unsupported = granularities - _TIMESTAMP_GRANULARITIES
    if unsupported:
        msg = f"unsupported timestamp granularities: {', '.join(sorted(unsupported))}"
        raise ValueError(msg)
    return granularities


def _waveform(payload: AudioPayload) -> np.ndarray:
    return np.frombuffer(payload.pcm_s16le, dtype="<i2").astype(np.float32) / 32_768.0


def _pad(waveform: np.ndarray, sample_rate: int, leading_s: float, trailing_s: float) -> np.ndarray:
    leading = np.zeros(round(leading_s * sample_rate), dtype=np.float32)
    trailing = np.zeros(round(trailing_s * sample_rate), dtype=np.float32)
    return np.concatenate([leading, waveform, trailing])


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


def _words_from_tokens(tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge timed tokens into words: a token that starts with a space opens a new word."""
    merged: list[dict[str, Any]] = []
    for token in tokens:
        piece = token["token"]
        if not merged or piece.startswith(" "):
            merged.append({"word": piece, "start": token["start"], "end": token["end"]})
        else:
            merged[-1]["word"] += piece
            merged[-1]["end"] = max(merged[-1]["end"], token["end"])
    words = []
    for word in merged:
        text = word["word"].strip()
        if text:
            words.append({"word": text, "start": round(word["start"], 3), "end": round(word["end"], 3)})
    return words


def _segments_from_words(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split words into segments at sentence-final punctuation or after 30 seconds."""
    segments: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for word in words:
        current.append(word)
        if _SENTENCE_END_RE.search(word["word"]) or word["end"] - current[0]["start"] >= _MAX_SEGMENT_S:
            segments.append(_segment(current, len(segments)))
            current = []
    if current:
        segments.append(_segment(current, len(segments)))
    return segments


def _segment(words: list[dict[str, Any]], index: int) -> dict[str, Any]:
    return {
        "id": index,
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "text": " ".join(word["word"] for word in words),
    }
