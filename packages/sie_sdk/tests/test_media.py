from pathlib import Path

import pytest
from sie_sdk.media import convert_item_media, infer_media_format, to_media_bytes


def test_infer_media_format_from_suffix() -> None:
    assert infer_media_format("recording.WAV") == "wav"
    assert infer_media_format(Path("clip.webm")) == "webm"
    assert infer_media_format("recording") is None


def test_to_media_bytes_passes_bytes_through() -> None:
    assert to_media_bytes(b"audio", kind="audio") == (b"audio", None)


def test_to_media_bytes_reads_path_and_infers_format(tmp_path: Path) -> None:
    recording = tmp_path / "recording.flac"
    recording.write_bytes(b"audio")

    assert to_media_bytes(recording, kind="audio") == (b"audio", "flac")


def test_to_media_bytes_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        to_media_bytes(tmp_path / "missing.wav", kind="audio")


def test_convert_item_media_converts_direct_inputs(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"video")
    item = {"audio": b"audio", "video": clip}

    result = convert_item_media(item)

    assert result is item
    assert result["audio"] == {"data": b"audio", "format": None}
    assert result["video"] == {"data": b"video", "format": "mp4"}


def test_convert_item_media_preserves_explicit_metadata(tmp_path: Path) -> None:
    recording = tmp_path / "recording.wav"
    recording.write_bytes(b"audio")
    item = {
        "audio": {
            "data": recording,
            "format": "pcm",
            "sample_rate": 16_000,
        }
    }

    result = convert_item_media(item)

    assert result["audio"] == {
        "data": b"audio",
        "format": "pcm",
        "sample_rate": 16_000,
    }


def test_convert_item_media_requires_data_in_mapping() -> None:
    with pytest.raises(ValueError, match="Audio input must contain a 'data' field"):
        convert_item_media({"audio": {"format": "wav"}})
