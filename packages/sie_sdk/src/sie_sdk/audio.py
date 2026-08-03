"""Audio conversion utilities for SIE SDK.

Wire format: encoded audio bytes plus a format hint. NumPy waveforms are
serialized as PCM WAV so the server receives the same self-describing payload
as it does for file-based inputs.
"""

from __future__ import annotations

import io
import wave
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

AudioLike = bytes | str | Path | NDArray[Any]

_SUFFIX_TO_FORMAT: dict[str, str] = {
    ".flac": "flac",
    ".m4a": "m4a",
    ".mp3": "mp3",
    ".mp4": "mp4",
    ".mpeg": "mpeg",
    ".mpga": "mpga",
    ".ogg": "ogg",
    ".wav": "wav",
    ".webm": "webm",
}


def infer_audio_format(source: str | Path) -> str | None:
    """Infer an audio format hint from a path suffix."""
    return _SUFFIX_TO_FORMAT.get(Path(source).suffix.lower())


def _waveform_to_wav_bytes(audio: NDArray[Any], sample_rate: int) -> bytes:
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        msg = "Audio sample_rate must be a positive integer"
        raise ValueError(msg)

    waveform = np.asarray(audio)
    if waveform.ndim not in (1, 2):
        msg = f"Expected a 1D mono or 2D audio array, got {waveform.ndim}D"
        raise ValueError(msg)

    if waveform.ndim == 2:
        # Accept the conventional [samples, channels] layout. The transposed
        # [channels, samples] form is unambiguous for mono/stereo audio.
        if waveform.shape[0] <= 2 < waveform.shape[1]:
            waveform = waveform.T
        channels = waveform.shape[1]
    else:
        channels = 1
    if channels <= 0 or channels > 2:
        msg = f"Audio array must contain 1 or 2 channels, got {channels}"
        raise ValueError(msg)
    if waveform.size == 0:
        raise ValueError("Audio array must not be empty")

    if np.issubdtype(waveform.dtype, np.floating):
        if not np.isfinite(waveform).all():
            raise ValueError("Audio array must contain only finite samples")
        normalized = np.clip(waveform.astype(np.float64), -1.0, 1.0)
        scale = np.where(normalized < 0, 32768.0, 32767.0)
        pcm = np.rint(normalized * scale).astype("<i2")
    elif waveform.dtype == np.int16:
        pcm = waveform.astype("<i2", copy=False)
    elif np.issubdtype(waveform.dtype, np.signedinteger):
        info = np.iinfo(waveform.dtype)
        normalized = waveform.astype(np.float64) / float(-info.min)
        pcm = np.clip(np.rint(normalized * 32768.0), -32768, 32767).astype("<i2")
    elif np.issubdtype(waveform.dtype, np.unsignedinteger):
        info = np.iinfo(waveform.dtype)
        midpoint = float(info.max // 2 + 1)
        normalized = (waveform.astype(np.float64) - midpoint) / midpoint
        pcm = np.clip(np.rint(normalized * 32768.0), -32768, 32767).astype("<i2")
    else:
        msg = f"Unsupported audio array dtype: {waveform.dtype}"
        raise TypeError(msg)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes(order="C"))
    return buffer.getvalue()


def to_audio_bytes(audio: AudioLike, *, sample_rate: int | None = None) -> tuple[bytes, str | None]:
    """Resolve an audio input to encoded bytes and an optional format hint."""
    if isinstance(audio, bytes):
        return audio, None

    if isinstance(audio, (str, Path)):
        path = Path(audio)
        if not path.exists():
            msg = f"Audio file not found: {path}"
            raise FileNotFoundError(msg)
        return path.read_bytes(), infer_audio_format(path)

    if isinstance(audio, np.ndarray):
        if sample_rate is None:
            raise ValueError("Audio sample_rate is required when data is a NumPy array")
        return _waveform_to_wav_bytes(audio, sample_rate), "wav"

    msg = f"Unsupported audio type: {type(audio)}. Expected bytes, NumPy array, str, or Path."
    raise TypeError(msg)


def convert_item_audio(item: dict[str, Any]) -> dict[str, Any]:
    """Convert an item's audio field to the server wire format in-place."""
    if "audio" not in item or item["audio"] is None:
        return item

    audio = item["audio"]
    if isinstance(audio, dict) and "data" in audio:
        data, inferred = to_audio_bytes(audio["data"], sample_rate=audio.get("sample_rate"))
        fmt = audio.get("format", inferred)
        item["audio"] = {
            "data": data,
            "format": fmt,
            "sample_rate": audio.get("sample_rate"),
        }
        return item

    data, inferred = to_audio_bytes(audio)
    item["audio"] = {"data": data, "format": inferred, "sample_rate": None}
    return item
