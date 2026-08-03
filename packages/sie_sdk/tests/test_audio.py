import io
import wave
from pathlib import Path

import numpy as np
import pytest
from sie_sdk.audio import convert_item_audio, infer_audio_format, to_audio_bytes


def _decode_wav(data: bytes) -> tuple[int, int, int, np.ndarray]:
    with wave.open(io.BytesIO(data), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        frame_count = wav.getnframes()
        samples = np.frombuffer(wav.readframes(frame_count), dtype="<i2").reshape(frame_count, channels)
    return sample_rate, channels, frame_count, samples


class TestInferAudioFormat:
    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("clip.wav", "wav"),
            ("CLIP.MP3", "mp3"),
            ("speech.flac", "flac"),
            ("recording.m4a", "m4a"),
            ("video.webm", "webm"),
        ],
    )
    def test_known_suffixes(self, filename: str, expected: str) -> None:
        assert infer_audio_format(filename) == expected
        assert infer_audio_format(Path(filename)) == expected

    def test_unknown_suffix_returns_none(self) -> None:
        assert infer_audio_format("clip.bin") is None


class TestToAudioBytes:
    def test_bytes_passthrough(self) -> None:
        payload = b"RIFF fake wav"
        data, fmt = to_audio_bytes(payload)
        assert data is payload
        assert fmt is None

    def test_path_input_reads_bytes_and_infers_format(self, tmp_path: Path) -> None:
        path = tmp_path / "clip.mp3"
        path.write_bytes(b"ID3 fake mp3")
        data, fmt = to_audio_bytes(path)
        assert data == b"ID3 fake mp3"
        assert fmt == "mp3"

    def test_string_path_input(self, tmp_path: Path) -> None:
        path = tmp_path / "clip.wav"
        path.write_bytes(b"RIFF fake wav")
        data, fmt = to_audio_bytes(str(path))
        assert data == b"RIFF fake wav"
        assert fmt == "wav"

    def test_missing_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            to_audio_bytes(tmp_path / "missing.wav")

    def test_invalid_input_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Unsupported audio type"):
            to_audio_bytes(123)  # ty: ignore[invalid-argument-type]

    def test_array_requires_sample_rate(self) -> None:
        with pytest.raises(ValueError, match="sample_rate is required"):
            to_audio_bytes(np.zeros(8, dtype=np.float32))

    def test_float_waveform_is_encoded_as_pcm16_wav(self) -> None:
        audio = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5], dtype=np.float32)

        data, fmt = to_audio_bytes(audio, sample_rate=16_000)

        sample_rate, channels, frame_count, samples = _decode_wav(data)
        assert fmt == "wav"
        assert sample_rate == 16_000
        assert channels == 1
        assert frame_count == 7
        np.testing.assert_array_equal(
            samples[:, 0],
            np.array([-32768, -32768, -16384, 0, 16384, 32767, 32767], dtype=np.int16),
        )

    @pytest.mark.parametrize(
        ("audio", "expected"),
        [
            (
                np.array([-32768, -1234, 0, 1234, 32767], dtype=np.int16),
                np.array([-32768, -1234, 0, 1234, 32767], dtype=np.int16),
            ),
            (
                np.array([-128, -64, 0, 64, 127], dtype=np.int8),
                np.array([-32768, -16384, 0, 16384, 32512], dtype=np.int16),
            ),
            (
                np.array([0, 64, 128, 192, 255], dtype=np.uint8),
                np.array([-32768, -16384, 0, 16384, 32512], dtype=np.int16),
            ),
        ],
    )
    def test_integer_waveform_is_scaled_to_pcm16(self, audio: np.ndarray, expected: np.ndarray) -> None:
        data, _ = to_audio_bytes(audio, sample_rate=44_100)

        sample_rate, channels, frame_count, samples = _decode_wav(data)
        assert sample_rate == 44_100
        assert channels == 1
        assert frame_count == len(audio)
        np.testing.assert_array_equal(samples[:, 0], expected)

    @pytest.mark.parametrize("channels_first", [False, True])
    def test_stereo_layouts_are_interleaved_by_frame(self, channels_first: bool) -> None:
        samples_first = np.column_stack(
            (
                np.linspace(-1.0, 1.0, 10, dtype=np.float32),
                np.linspace(1.0, -1.0, 10, dtype=np.float32),
            )
        )
        audio = samples_first.T if channels_first else samples_first

        data, _ = to_audio_bytes(audio, sample_rate=22_050)

        sample_rate, channels, frame_count, samples = _decode_wav(data)
        assert sample_rate == 22_050
        assert channels == 2
        assert frame_count == 10
        assert samples[0].tolist() == [-32768, 32767]
        assert samples[-1].tolist() == [32767, -32768]

    @pytest.mark.parametrize("sample_rate", [True, 0, -1, 16_000.0])
    def test_array_rejects_invalid_sample_rate(self, sample_rate: object) -> None:
        with pytest.raises(ValueError, match="sample_rate must be a positive integer"):
            to_audio_bytes(
                np.zeros(8, dtype=np.float32),
                sample_rate=sample_rate,  # ty: ignore[invalid-argument-type]
            )

    @pytest.mark.parametrize(
        "audio",
        [
            np.array(0.0, dtype=np.float32),
            np.zeros((2, 3, 4), dtype=np.float32),
        ],
    )
    def test_array_rejects_invalid_dimensions(self, audio: np.ndarray) -> None:
        with pytest.raises(ValueError, match="Expected a 1D mono or 2D audio array"):
            to_audio_bytes(audio, sample_rate=16_000)

    @pytest.mark.parametrize(
        "audio",
        [
            np.array([], dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
        ],
    )
    def test_array_rejects_empty_input(self, audio: np.ndarray) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            to_audio_bytes(audio, sample_rate=16_000)

    @pytest.mark.parametrize("sample", [np.nan, np.inf, -np.inf])
    def test_array_rejects_nonfinite_samples(self, sample: float) -> None:
        with pytest.raises(ValueError, match="only finite samples"):
            to_audio_bytes(np.array([sample], dtype=np.float32), sample_rate=16_000)

    def test_array_rejects_more_than_two_channels(self) -> None:
        with pytest.raises(ValueError, match="1 or 2 channels"):
            to_audio_bytes(np.zeros((10, 9), dtype=np.float32), sample_rate=16_000)

    @pytest.mark.parametrize(
        "audio",
        [
            np.array([True, False]),
            np.array([1 + 2j], dtype=np.complex64),
        ],
    )
    def test_array_rejects_unsupported_dtype(self, audio: np.ndarray) -> None:
        with pytest.raises(TypeError, match="Unsupported audio array dtype"):
            to_audio_bytes(audio, sample_rate=16_000)


class TestConvertItemAudio:
    def test_no_audio_field_returns_unchanged(self) -> None:
        item = {"text": "hello"}
        assert convert_item_audio(item) is item
        assert item == {"text": "hello"}

    def test_bytes_input_is_normalized(self) -> None:
        item = {"audio": b"RIFF fake wav"}
        convert_item_audio(item)
        assert item["audio"] == {
            "data": b"RIFF fake wav",
            "format": None,
            "sample_rate": None,
        }

    def test_path_input_infers_format(self, tmp_path: Path) -> None:
        path = tmp_path / "clip.flac"
        path.write_bytes(b"fLaC fake")
        item = {"audio": path}
        convert_item_audio(item)
        assert item["audio"] == {
            "data": b"fLaC fake",
            "format": "flac",
            "sample_rate": None,
        }

    def test_dict_preserves_explicit_metadata(self, tmp_path: Path) -> None:
        path = tmp_path / "clip.wav"
        path.write_bytes(b"RIFF fake wav")
        item = {
            "audio": {
                "data": path,
                "format": "wave",
                "sample_rate": 48_000,
            }
        }
        convert_item_audio(item)
        assert item["audio"] == {
            "data": b"RIFF fake wav",
            "format": "wave",
            "sample_rate": 48_000,
        }

    def test_explicit_none_metadata_is_preserved(self, tmp_path: Path) -> None:
        path = tmp_path / "clip.wav"
        path.write_bytes(b"RIFF fake wav")
        item = {"audio": {"data": path, "format": None, "sample_rate": None}}
        convert_item_audio(item)
        assert item["audio"] == {
            "data": b"RIFF fake wav",
            "format": None,
            "sample_rate": None,
        }
