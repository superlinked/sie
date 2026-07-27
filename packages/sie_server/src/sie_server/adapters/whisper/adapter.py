"""Whisper speech-to-text adapter implemented on the native extract primitive."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ComputePrecision
from sie_server.core.inference_output import EncodeOutput, ExtractOutput
from sie_server.core.prepared import AudioPayload
from sie_server.core.preprocessor.audio import AudioPreprocessor

if TYPE_CHECKING:
    from sie_server.types.inputs import Item

logger = logging.getLogger(__name__)

_ERR_ENCODE_NOT_SUPPORTED = "WhisperAdapter does not support encode(). Use extract() instead."
_TIMESTAMP_GRANULARITIES = frozenset({"segment", "word"})
_RUNTIME_OPTIONS = frozenset({"language", "temperature", "timestamp_granularities"})
_SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]*$")


class WhisperAdapter(BaseAdapter):
    """Batched long-form transcription for Whisper large-v3-turbo."""

    spec: ClassVar[AdapterSpec] = AdapterSpec(
        inputs=("audio",),
        outputs=("json",),
        unload_fields=("_model", "_processor", "_pipeline", "_preprocessor"),
        default_preprocessor="audio",
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        compute_precision: ComputePrecision = "float16",
        attn_implementation: str = "sdpa",
        revision: str | None = None,
        pipeline_batch_size: int = 8,
        **kwargs: Any,
    ) -> None:
        del kwargs
        if pipeline_batch_size <= 0:
            msg = "pipeline_batch_size must be positive"
            raise ValueError(msg)
        self._model_name_or_path = str(model_name_or_path)
        self._compute_precision = compute_precision
        self._attn_implementation = attn_implementation
        self._revision = revision
        self._pipeline_batch_size = pipeline_batch_size
        self._model: Any = None
        self._processor: Any = None
        self._pipeline: Any = None
        self._preprocessor: AudioPreprocessor | None = None
        self._device: str | None = None

    def load(self, device: str) -> None:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        self._device = device
        dtype = self._resolve_dtype()
        shared_kwargs: dict[str, Any] = {}
        if self._revision is not None:
            shared_kwargs["revision"] = self._revision

        logger.info(
            "Loading Whisper model %s on device=%s with dtype=%s, attn=%s",
            self._model_name_or_path,
            device,
            dtype,
            self._attn_implementation,
        )
        self._processor = AutoProcessor.from_pretrained(
            self._model_name_or_path,
            **shared_kwargs,
        )
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self._model_name_or_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation=self._attn_implementation,
            **shared_kwargs,
        )
        self._model.to(device)
        self._model.eval()
        self._pipeline = pipeline(
            "automatic-speech-recognition",
            model=self._model,
            tokenizer=self._processor.tokenizer,
            feature_extractor=self._processor.feature_extractor,
            torch_dtype=dtype,
            device=device,
        )
        self._preprocessor = AudioPreprocessor()

    def _resolve_dtype(self) -> torch.dtype:
        if not self._device or not self._device.startswith("cuda"):
            return torch.float32
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(self._compute_precision, torch.float16)

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
        if self._pipeline is None or self._processor is None:
            msg = "WhisperAdapter is not loaded"
            raise RuntimeError(msg)
        if labels:
            msg = "Whisper transcription does not accept entity labels"
            raise ValueError(msg)
        if output_schema:
            msg = "Whisper transcription does not accept output_schema"
            raise ValueError(msg)

        payloads = _audio_payloads(items, prepared_items)
        language, temperature, granularities = _parse_options(options)

        generation_kwargs: dict[str, Any] = {}
        if language is not None:
            generation_kwargs["language"] = language
        if temperature is not None:
            generation_kwargs["temperature"] = temperature
        if instruction:
            prompt_ids = self._processor.get_prompt_ids(
                instruction,
                return_tensors="pt",
            )
            if not isinstance(prompt_ids, torch.Tensor):
                msg = "Whisper processor returned invalid prompt IDs"
                raise TypeError(msg)
            generation_kwargs["prompt_ids"] = prompt_ids.to(self._device)

        pipeline_inputs = [
            {
                "raw": np.frombuffer(payload.pcm_s16le, dtype="<i2").astype(np.float32) / 32_768.0,
                "sampling_rate": payload.sample_rate,
            }
            for payload in payloads
        ]
        raw_outputs: list[dict[str, Any] | None] = [None] * len(payloads)
        max_samples = self._processor.feature_extractor.n_samples
        duration_groups = (
            ([index for index, payload in enumerate(payloads) if payload.sample_count <= max_samples], False),
            ([index for index, payload in enumerate(payloads) if payload.sample_count > max_samples], True),
        )
        for indices, long_form in duration_groups:
            if not indices:
                continue
            timestamp_mode: bool | str
            if "word" in granularities:
                timestamp_mode = "word"
            else:
                timestamp_mode = long_form or "segment" in granularities
            group_outputs = self._pipeline(
                [pipeline_inputs[index] for index in indices],
                batch_size=self._pipeline_batch_size,
                return_timestamps=timestamp_mode,
                return_language=True,
                generate_kwargs=generation_kwargs,
            )
            if isinstance(group_outputs, dict):
                group_outputs = [group_outputs]
            if len(group_outputs) != len(indices):
                msg = "Whisper pipeline returned a misaligned batch"
                raise RuntimeError(msg)
            for index, output in zip(indices, group_outputs, strict=True):
                raw_outputs[index] = output
        if any(output is None for output in raw_outputs):
            msg = "Whisper pipeline returned an incomplete batch"
            raise RuntimeError(msg)

        data = [
            _transcript_data(output, payload, granularities, language)
            for output, payload in zip(raw_outputs, payloads, strict=True)
            if output is not None
        ]
        return ExtractOutput(
            entities=[[] for _ in data],
            data=data,
            batch_size=len(data),
        )


def _audio_payloads(items: list[Item], prepared_items: list[Any] | None) -> list[AudioPayload]:
    if prepared_items is None or len(prepared_items) != len(items):
        msg = "WhisperAdapter requires one Rust-prepared audio payload per item"
        raise ValueError(msg)
    payloads = []
    for prepared in prepared_items:
        payload = getattr(prepared, "payload", None)
        if not isinstance(payload, AudioPayload):
            msg = "WhisperAdapter received a non-audio prepared item"
            raise TypeError(msg)
        payloads.append(payload)
    return payloads


def _parse_options(
    options: dict[str, Any] | None,
) -> tuple[str | None, float | None, frozenset[str]]:
    options = options or {}
    unknown = set(options) - _RUNTIME_OPTIONS
    if unknown:
        msg = f"unsupported Whisper options: {', '.join(sorted(unknown))}"
        raise ValueError(msg)

    language = options.get("language")
    if language is not None and (not isinstance(language, str) or not language.strip()):
        msg = "language must be a non-empty string or null"
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

    raw_granularities = options.get("timestamp_granularities", [])
    if not isinstance(raw_granularities, list) or not all(isinstance(value, str) for value in raw_granularities):
        msg = "timestamp_granularities must be a list of strings"
        raise ValueError(msg)
    granularities = frozenset(value for value in raw_granularities if isinstance(value, str))
    unknown_granularities = granularities - _TIMESTAMP_GRANULARITIES
    if unknown_granularities:
        msg = f"unsupported timestamp granularities: {', '.join(sorted(unknown_granularities))}"
        raise ValueError(msg)
    normalized_language: str | None = language.strip() if isinstance(language, str) else None
    return normalized_language, temperature, granularities


def _transcript_data(
    output: dict[str, Any],
    payload: AudioPayload,
    granularities: frozenset[str],
    language_hint: str | None,
) -> dict[str, Any]:
    chunks = output.get("chunks") or []
    detected_language = next(
        (chunk.get("language") for chunk in chunks if chunk.get("language")),
        language_hint,
    )
    data: dict[str, Any] = {
        "text": str(output.get("text", "")),
        "language": detected_language,
        "duration_ms": payload.duration_ms,
    }
    if "word" in granularities:
        words = [_word_timestamp_entry(chunk) for chunk in chunks]
        data["words"] = words
        if "segment" in granularities:
            data["segments"] = _segments_from_words(words)
    elif "segment" in granularities:
        data["segments"] = [_segment_timestamp_entry(chunk, index) for index, chunk in enumerate(chunks)]
    return data


def _word_timestamp_entry(chunk: dict[str, Any]) -> dict[str, Any]:
    timestamp = chunk.get("timestamp") or (None, None)
    return {
        "word": str(chunk.get("text", "")),
        "start": timestamp[0],
        "end": timestamp[1],
    }


def _segment_timestamp_entry(chunk: dict[str, Any], index: int) -> dict[str, Any]:
    timestamp = chunk.get("timestamp") or (None, None)
    return {
        "id": index,
        "start": timestamp[0],
        "end": timestamp[1],
        "text": str(chunk.get("text", "")),
    }


def _segments_from_words(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for word in words:
        current.append(word)
        start = current[0]["start"]
        end = word["end"]
        span = end - start if isinstance(start, (int, float)) and isinstance(end, (int, float)) else 0
        if _SENTENCE_END_RE.search(word["word"].strip()) or span >= 30:
            segments.append(_word_segment(current, len(segments)))
            current = []
    if current:
        segments.append(_word_segment(current, len(segments)))
    return segments


def _word_segment(words: list[dict[str, Any]], index: int) -> dict[str, Any]:
    return {
        "id": index,
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "text": "".join(word["word"] for word in words),
    }
