"""Qwen3-ASR transcription on SIE's prepared audio extraction primitive."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast

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

_RUNTIME_OPTIONS = frozenset({"language", "temperature", "max_new_tokens", "timestamp_granularities"})


class Qwen3ASRAdapter(BaseAdapter):
    """Transcribe complete prepared waveforms with the native Transformers processor."""

    spec: ClassVar[AdapterSpec] = AdapterSpec(
        inputs=("audio",),
        outputs=("json",),
        unload_fields=("_model", "_processor", "_preprocessor", "_context_limit"),
        default_preprocessor="audio",
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        compute_precision: ComputePrecision = "bfloat16",
        attn_implementation: str = "sdpa",
        revision: str | None = None,
        max_new_tokens: int = 512,
        inference_batch_size: int = 4,
        **kwargs: Any,
    ) -> None:
        del kwargs
        _positive_integer(max_new_tokens, "max_new_tokens")
        _positive_integer(inference_batch_size, "inference_batch_size")
        if compute_precision not in {"float16", "bfloat16", "float32"}:
            msg = "Qwen3-ASR supports float16, bfloat16 or float32 compute precision"
            raise ValueError(msg)
        self._model_name_or_path = str(model_name_or_path)
        self._compute_precision = compute_precision
        self._attn_implementation = attn_implementation
        self._revision = revision
        self._max_new_tokens = max_new_tokens
        self._inference_batch_size = inference_batch_size
        self._model: Any = None
        self._processor: Any = None
        self._preprocessor: AudioPreprocessor | None = None
        self._device: str | None = None
        self._context_limit: int | None = None

    def load(self, device: str) -> None:
        import transformers
        from transformers import AutoProcessor

        model_class = getattr(transformers, "AutoModelForMultimodalLM", None)
        if model_class is None:
            msg = "Qwen3-ASR requires the Transformers5 bundle"
            raise RuntimeError(msg)

        dtype = (
            torch.float32
            if not device.startswith("cuda")
            else {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }[self._compute_precision]
        )
        shared: dict[str, Any] = {"trust_remote_code": False}
        if self._revision is not None:
            shared["revision"] = self._revision
        self._processor = AutoProcessor.from_pretrained(self._model_name_or_path, **shared)
        if self._processor.feature_extractor.sampling_rate != 16_000:
            msg = "Qwen3-ASR requires the 16kHz processor used by prepared audio"
            raise ValueError(msg)
        self._model = model_class.from_pretrained(
            self._model_name_or_path,
            dtype=dtype,
            use_safetensors=True,
            attn_implementation=self._attn_implementation,
            **shared,
        )
        self._model.to(device)
        self._model.eval()
        self._context_limit = self._model.config.text_config.max_position_embeddings
        _positive_integer(self._context_limit, "model text context")
        self._device = device
        self._preprocessor = AudioPreprocessor()

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
        msg = "Qwen3ASRAdapter does not support encode(). Use extract() instead."
        raise NotImplementedError(msg)

    @torch.inference_mode()
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
        if self._processor is None or self._device is None or self._context_limit is None:
            msg = "Qwen3ASRAdapter is not loaded"
            raise RuntimeError(msg)
        if labels or output_schema:
            msg = "Qwen3-ASR transcription does not accept entity labels or output_schema"
            raise ValueError(msg)
        language, temperature, maximum = _parse_options(options, self._max_new_tokens)
        payloads = _audio_payloads(items, prepared_items)
        # Admit all CPU batches before moving one batch at a time to the model.
        prepared_batches = deque()
        for start in range(0, len(payloads), self._inference_batch_size):
            batch = payloads[start : start + self._inference_batch_size]
            waveforms = [
                np.frombuffer(payload.pcm_s16le, dtype="<i2").astype(np.float32) / 32_768.0 for payload in batch
            ]
            inputs = self._processor.apply_transcription_request(
                audio=waveforms,
                language=language,
                prompt=instruction,
                text_kwargs={"padding": True, "padding_side": "left"},
                audio_kwargs={
                    "sampling_rate": 16_000,
                    "padding": True,
                    "truncation": False,
                    "return_attention_mask": True,
                },
                return_tensors="pt",
            )
            if (
                inputs["input_ids"].ndim != 2
                or inputs["input_ids"].shape[0] != len(batch)
                or inputs["attention_mask"].shape != inputs["input_ids"].shape
            ):
                msg = "Qwen3-ASR processor returned misaligned input tokens or masks"
                raise RuntimeError(msg)
            width = inputs["input_ids"].shape[1]
            if width + maximum > self._context_limit:
                msg = "Qwen3-ASR prompt, batch padding and requested output exceed the model context"
                raise ValueError(msg)
            prepared_batches.append((batch, inputs, width))

        generation: dict[str, Any] = {
            "max_new_tokens": maximum,
            "do_sample": temperature > 0,
            "return_dict_in_generate": True,
        }
        if temperature > 0:
            generation["temperature"] = temperature
        data = []
        while prepared_batches:
            batch, inputs, width = prepared_batches.popleft()
            inputs = inputs.to(self._device, dtype=self._model.dtype)
            generated = self._model.generate(**inputs, **generation).sequences
            if (
                not isinstance(generated, torch.Tensor)
                or generated.ndim != 2
                or generated.shape[0] != len(batch)
                or generated.shape[1] < width
            ):
                msg = "Qwen3-ASR returned a misaligned generated batch"
                raise RuntimeError(msg)
            completions = generated[:, width:]
            parsed = self._processor.decode(completions, return_format="parsed")
            if not isinstance(parsed, list) or len(parsed) != len(batch):
                msg = "Qwen3-ASR processor returned a misaligned transcription batch"
                raise RuntimeError(msg)
            eos_ids = self._model.generation_config.eos_token_id
            for raw_result, payload, token_ids in zip(parsed, batch, completions.tolist(), strict=True):
                if not isinstance(raw_result, dict):
                    msg = "Qwen3-ASR processor returned invalid parsed transcription"
                    raise RuntimeError(msg)
                result = cast("dict[str, Any]", raw_result)
                text = result.get("transcription")
                result_language = result.get("language")
                if not isinstance(text, str) or (result_language is not None and not isinstance(result_language, str)):
                    msg = "Qwen3-ASR processor returned invalid parsed transcription"
                    raise RuntimeError(msg)
                count, finish, cap = _termination(token_ids, eos_ids, maximum)
                data.append(
                    {
                        "text": text,
                        "language": result_language,
                        "duration_ms": payload.duration_ms,
                        "output_tokens": count,
                        "max_new_tokens": maximum,
                        "finish_reason": finish,
                        "cap_state": cap,
                    }
                )
        return ExtractOutput(entities=[[] for _ in data], data=data, batch_size=len(data))


def _positive_integer(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        msg = f"{name} must be a positive integer"
        raise ValueError(msg)


def _audio_payloads(items: list[Item], prepared: list[Any] | None) -> list[AudioPayload]:
    if prepared is None or len(prepared) != len(items):
        msg = "Qwen3ASRAdapter requires one Rust-prepared audio payload per item"
        raise ValueError(msg)
    payloads = []
    for item in prepared:
        payload = getattr(item, "payload", None)
        if not isinstance(payload, AudioPayload):
            msg = "Qwen3ASRAdapter received a non-audio prepared item"
            raise TypeError(msg)
        if (
            payload.sample_rate != 16_000
            or payload.sample_count <= 0
            or len(payload.pcm_s16le) != 2 * payload.sample_count
        ):
            msg = "Qwen3-ASR requires complete mono 16kHz s16le prepared audio"
            raise ValueError(msg)
        payloads.append(payload)
    return payloads


def _parse_options(options: dict[str, Any] | None, default_maximum: int) -> tuple[str | None, float, int]:
    options = options or {}
    unknown = set(options) - _RUNTIME_OPTIONS
    if unknown:
        msg = f"unsupported Qwen3-ASR options: {', '.join(sorted(unknown))}"
        raise ValueError(msg)
    granularities = options.get("timestamp_granularities", [])
    if not isinstance(granularities, list) or granularities:
        msg = "Qwen3-ASR does not provide word or segment timestamps; a separate forced aligner is required"
        raise ValueError(msg)
    language = options.get("language")
    if language is not None and (not isinstance(language, str) or not language.strip()):
        msg = "language must be a non-empty string or null"
        raise ValueError(msg)
    temperature = options.get("temperature", 0.0)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= temperature <= 1:
        msg = "temperature must be a number between 0 and 1"
        raise ValueError(msg)
    maximum = options.get("max_new_tokens", default_maximum)
    _positive_integer(maximum, "max_new_tokens")
    return language, float(temperature), maximum


def _termination(tokens: list[int], eos_ids: int | list[int] | None, maximum: int) -> tuple[int, str | None, str]:
    eos = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
    for index, token in enumerate(tokens):
        if token in eos:
            return index + 1, "stop", "observed_stop"
    if len(tokens) == maximum:
        return len(tokens), "length", "capped_as_returned"
    return len(tokens), None, "unknown"
