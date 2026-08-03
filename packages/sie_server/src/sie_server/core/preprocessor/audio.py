"""Canonical audio preprocessing for direct, non-sidecar execution."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Protocol, cast

from sie_server.core.prepared import AudioPayload, PreparedBatch, PreparedItem
from sie_server.types.inputs import media_bytes

if TYPE_CHECKING:
    from sie_server.config.model import ModelConfig
    from sie_server.types.inputs import Item


class _AudioPrepModule(Protocol):
    def decode_audio(self, data: bytes, format: str | None = None) -> dict[str, Any]:
        raise NotImplementedError


class AudioPreprocessor:
    """Decode and resample audio through the shared Rust extension."""

    @property
    def modality(self) -> str:
        return "audio"

    def prepare(
        self,
        items: list[Item],
        *,
        config: ModelConfig,
        is_query: bool = False,
        instruction: str | None = None,
        task: str | None = None,
    ) -> PreparedBatch[AudioPayload]:
        try:
            audio_prep = cast("_AudioPrepModule", importlib.import_module("sie_audio_prep"))
        except ModuleNotFoundError as exc:
            if exc.name != "sie_audio_prep":
                raise
            msg = "Audio preprocessing requires a SIE image or workspace build containing sie-audio-prep"
            raise RuntimeError(msg) from exc
        prepared_items: list[PreparedItem[AudioPayload]] = []
        total_cost = 0

        for index, item in enumerate(items):
            if item.audio is None:
                msg = "audio extract items must include audio"
                raise ValueError(msg)
            audio = item.audio
            encoded = media_bytes(audio, kind="audio")
            declared_format = audio.get("format")
            if declared_format is not None and not isinstance(declared_format, str):
                msg = "audio.format must be a string or null"
                raise ValueError(msg)
            decoded = audio_prep.decode_audio(encoded, declared_format)
            declared_sample_rate = audio.get("sample_rate")
            if declared_sample_rate is not None:
                if (
                    not isinstance(declared_sample_rate, int)
                    or isinstance(declared_sample_rate, bool)
                    or declared_sample_rate <= 0
                ):
                    msg = "audio.sample_rate must be a positive integer or null"
                    raise ValueError(msg)
                if declared_sample_rate != decoded["source_sample_rate"]:
                    msg = (
                        f"declared audio sample_rate {declared_sample_rate} does not match "
                        f"decoded {decoded['source_sample_rate']} Hz"
                    )
                    raise ValueError(msg)

            payload = AudioPayload(
                pcm_s16le=decoded["pcm_s16le"],
                sample_rate=decoded["sample_rate"],
                sample_count=decoded["sample_count"],
                duration_ms=decoded["duration_ms"],
                source_sample_rate=decoded["source_sample_rate"],
                source_sample_count=decoded["source_sample_count"],
                source_channels=decoded["source_channels"],
                container=decoded["container"],
            )
            prepared_items.append(
                PreparedItem(
                    payload=payload,
                    cost=payload.duration_cost_ms,
                    original_index=index,
                )
            )
            total_cost += payload.duration_cost_ms

        return PreparedBatch(items=prepared_items, total_cost=total_cost, modality="audio")

    def collate(
        self,
        prepared: list[PreparedItem[AudioPayload]],
        *,
        device: str,
    ) -> dict[str, Any]:
        return {"audio": [item.payload for item in prepared]}
