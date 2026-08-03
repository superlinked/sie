"""Video frame extraction seam: the basis on which video bills as images (§7).

Every test video is GENERATED here with the same OpenCV that decodes it — no
binary fixture is downloaded or committed. The default fixture uses the
lossless FFV1 codec and fills frame N with grey level ``N * 8``, so a sampled
frame's pixel value identifies its source index exactly and the sampling stride
is directly assertable. Lossy containers (mp4/webm) get their own count-only
tests, since they are the formats the removed placeholder silently dropped.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sie_server.core import video_frames
from sie_server.core.video_frames import (
    MAX_SAMPLED_FRAMES,
    VideoDecodeError,
    extract_frames,
)
from sie_server.types.inputs import InvalidInputError, InvalidMediaError

cv2 = pytest.importorskip("cv2", reason="video decoding requires the OpenCV wheel")

_GREY_STEP = 8


def _make_video(
    tmp_path: Path,
    *,
    frames: int,
    fps: float = 10.0,
    codec: str = "FFV1",
    suffix: str = ".avi",
) -> bytes:
    """Encode a tiny video whose frame N is uniformly filled with grey ``N * 8``.

    The grey level wraps past 31 frames; ``_source_indices`` is therefore only
    meaningful for the short fixtures, and the long ones assert counts.

    Encoder availability varies by ``opencv-python-headless`` wheel and base
    image, so a codec this wheel cannot write skips rather than fails — same
    contract as the module-level ``importorskip("cv2")``. The billing
    invariants that must never silently lapse (the budget-fill arithmetic and
    the decode-work bounds) are asserted against fake captures instead and do
    not depend on any encoder being present.
    """
    path = tmp_path / f"generated-{codec}{suffix}"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (32, 24))
    if not writer.isOpened():
        writer.release()
        pytest.skip(f"this OpenCV wheel cannot encode {codec}{suffix}")
    for index in range(frames):
        writer.write(np.full((24, 32, 3), (index * _GREY_STEP) % 256, dtype=np.uint8))
    writer.release()
    return path.read_bytes()


def _video(data: bytes, fmt: str | None = "avi") -> dict[str, Any]:
    return {"data": data, "format": fmt}


def _source_indices(frames: list[Any]) -> list[int]:
    """Recover each sampled frame's index in the source stream."""
    return [int(np.asarray(frame)[0, 0, 0]) // _GREY_STEP for frame in frames]


class TestExtractFrames:
    def test_decodes_real_mp4_into_frames(self, tmp_path: Path) -> None:
        # The placeholder this replaces returned [] for every real container.
        data = _make_video(tmp_path, frames=8, codec="mp4v", suffix=".mp4")
        frames = extract_frames(_video(data, "mp4"))
        assert len(frames) == 8
        assert frames[0].mode == "RGB"
        assert frames[0].size == (32, 24)

    def test_decodes_real_webm_into_frames(self, tmp_path: Path) -> None:
        data = _make_video(tmp_path, frames=6, codec="VP80", suffix=".webm")
        assert len(extract_frames(_video(data, "webm"))) == 6

    def test_samples_uniformly_under_the_budget(self, tmp_path: Path) -> None:
        # 20 source frames, budget 4 -> evenly spaced indices 0,5,10,15.
        frames = extract_frames(_video(_make_video(tmp_path, frames=20)), max_frames=4)
        assert _source_indices(frames) == [0, 5, 10, 15]

    def test_never_exceeds_the_billing_budget(self, tmp_path: Path) -> None:
        # The managed reservation ceiling is derived from MAX_SAMPLED_FRAMES, so
        # a longer video must still settle at or below it.
        frames = extract_frames(_video(_make_video(tmp_path, frames=MAX_SAMPLED_FRAMES * 3)))
        assert len(frames) <= MAX_SAMPLED_FRAMES

    def test_a_caller_budget_cannot_raise_the_billing_cap(self, tmp_path: Path) -> None:
        frames = extract_frames(
            _video(_make_video(tmp_path, frames=MAX_SAMPLED_FRAMES * 2)),
            max_frames=MAX_SAMPLED_FRAMES * 2,
        )
        assert len(frames) <= MAX_SAMPLED_FRAMES

    def test_short_video_yields_every_frame_once(self, tmp_path: Path) -> None:
        frames = extract_frames(_video(_make_video(tmp_path, frames=3)))
        assert _source_indices(frames) == [0, 1, 2]

    def test_sampling_is_deterministic(self, tmp_path: Path) -> None:
        # Same bytes -> same frames -> same bill, on every retry and replica.
        data = _make_video(tmp_path, frames=17)
        assert _source_indices(extract_frames(_video(data), max_frames=5)) == _source_indices(
            extract_frames(_video(data), max_frames=5)
        )

    def test_unknown_format_hint_still_decodes(self, tmp_path: Path) -> None:
        # The hint is advisory; the demuxer sniffs the container.
        assert extract_frames(_video(_make_video(tmp_path, frames=4), "quicktimeish"))

    def test_missing_format_hint_still_decodes(self, tmp_path: Path) -> None:
        assert extract_frames({"data": _make_video(tmp_path, frames=4)})

    def test_counts_frames_itself_when_metadata_is_unusable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Containers that report no fps/frame count fall back to a grab-only
        # counting pass; sampling must stay uniform over the real length.
        monkeypatch.setattr(video_frames, "_validated_frame_total", lambda *_: None)
        frames = extract_frames(_video(_make_video(tmp_path, frames=20)), max_frames=4)
        assert _source_indices(frames) == [0, 5, 10, 15]


class TestBudgetIsFilled:
    """A stream at or above the budget must yield exactly the budget.

    The uniform stride this replaces (``ceil(total / budget)``, retained where
    ``index % stride == 0``) under-filled it for nearly every length: 33 source
    frames at budget 32 yielded 17, 48 yielded 24, 100 yielded 25. That halves
    both the visual context the model sees and the customer's bill, at a
    boundary invisible to them — a 32-frame clip billed 32, a 33-frame clip 17.
    """

    @pytest.mark.parametrize("total", [33, 48, 63, 100])
    def test_every_stream_length_fills_the_budget(self, tmp_path: Path, total: int) -> None:
        frames = extract_frames(_video(_make_video(tmp_path, frames=total)))
        assert len(frames) == MAX_SAMPLED_FRAMES

    @pytest.mark.parametrize("total", [32, 33, 48, 63, 100, 1_000, 108_000])
    def test_indices_are_distinct_ordered_and_in_range(self, total: int) -> None:
        indices = video_frames._target_indices(total, MAX_SAMPLED_FRAMES)
        assert len(indices) == MAX_SAMPLED_FRAMES
        assert indices == sorted(set(indices))  # distinct and in stream order
        assert indices[0] == 0
        assert indices[-1] < total

    @pytest.mark.parametrize("total", [1, 5, 31])
    def test_a_stream_under_the_budget_yields_every_frame(self, total: int) -> None:
        assert video_frames._target_indices(total, MAX_SAMPLED_FRAMES) == list(range(total))


class _CountingCapture:
    """Seekable capture that records exactly how much decode work was demanded.

    Frame N is filled with grey ``N * 8`` like the generated fixtures, so
    ``_source_indices`` recovers which frames a sampling strategy actually took.
    """

    def __init__(self, total: int, *, seekable: bool = True) -> None:
        self.total = total
        self.seekable = seekable
        self.reads = 0
        self.grabs = 0
        self._index = 0

    def set(self, _prop: int, value: float) -> bool:
        if not self.seekable:
            return False
        self._index = int(value)
        return True

    def get(self, _prop: int) -> float:
        return float(self._index)

    def grab(self) -> bool:
        if self._index >= self.total:
            return False
        self._index += 1
        self.grabs += 1
        return True

    def read(self) -> tuple[bool, Any]:
        if self._index >= self.total:
            return False, None
        frame = np.full((2, 2, 3), (self._index * _GREY_STEP) % 256, dtype=np.uint8)
        self._index += 1
        self.reads += 1
        return True, frame


class _LyingCapture(_CountingCapture):
    """Backend whose ``set`` claims success but never moves the capture."""

    def set(self, _prop: int, value: float) -> bool:
        _ = value
        return True


class _KeyframeSnappingCapture(_CountingCapture):
    """Backend that snaps a seek back to the preceding keyframe, and reports it.

    OpenCV's FFmpeg backend documents exactly this for RAW mode (a request for
    frame ``i`` seeks to keyframe ``k <= i``), and wheels differ in whether the
    reported position is the request or the landing.
    """

    GOP = 8

    def set(self, _prop: int, value: float) -> bool:
        self._index = (int(value) // self.GOP) * self.GOP
        return True


class _OvershootingCapture(_CountingCapture):
    """Backend that lands PAST the requested frame."""

    def set(self, _prop: int, value: float) -> bool:
        self._index = min(self.total, int(value) + 5)
        return True


class _FakeCv2:
    CAP_PROP_POS_FRAMES = 1


class TestDecodeWorkIsBoundedByTheBudget:
    """Sampling cost must be the frame budget, not the stream length.

    ``extract_frames`` is called from inside ``adapter.encode``, which on the
    batched worker path runs on ``ModelWorker._inference_executor`` — a
    single-worker pool that serializes the model's GPU work. Walking a long
    stream there stalls every co-scheduled request in the container, including
    other tenants', until they time out at the gateway.
    """

    def test_seeking_decodes_only_the_sampled_frames(self) -> None:
        # 30 minutes at 30 fps: the walk this replaces issued ~54,000 grab/read
        # calls on the inference thread for a single item.
        capture = _CountingCapture(54_000)
        frames = video_frames._sample(_FakeCv2, capture, total=54_000, budget=MAX_SAMPLED_FRAMES)
        assert len(frames) == MAX_SAMPLED_FRAMES
        assert capture.reads == MAX_SAMPLED_FRAMES
        assert capture.grabs == 0  # not one frame of the stream was walked

    def test_a_backend_that_cannot_seek_still_samples_correctly(self) -> None:
        capture = _CountingCapture(40, seekable=False)
        frames = video_frames._sample(_FakeCv2, capture, total=40, budget=4)
        assert _source_indices(frames) == [0, 10, 20, 30]
        assert capture.grabs == 27  # three gaps of nine skipped frames

    def test_a_seek_that_lies_degrades_to_the_walk_instead_of_wrong_frames(self) -> None:
        # Trusting a backend that reports success without moving would sample
        # the wrong frames (a stuck position repeats frame 0 N times) — a
        # correctness failure, not merely a slow one. The cursor is resynced
        # from the reported position, so the walk still lands on target.
        capture = _LyingCapture(40)
        frames = video_frames._sample(_FakeCv2, capture, total=40, budget=4)
        assert _source_indices(frames) == [0, 10, 20, 30]
        assert capture.grabs == 27

    def test_a_keyframe_snapped_seek_resyncs_the_cursor(self) -> None:
        # A backend that lands on the preceding keyframe leaves the capture
        # somewhere the caller did not ask for. Walking on from a stale logical
        # index would overshoot and decode the WRONG frame (here: 17 instead of
        # 10), so the cursor is seeded from where the capture actually landed
        # and the short remainder is walked.
        capture = _KeyframeSnappingCapture(40)
        frames = video_frames._sample(_FakeCv2, capture, total=40, budget=4)
        assert _source_indices(frames) == [0, 10, 20, 30]
        # Only the sub-GOP remainders are walked: 8->10, 16->20, 24->30.
        assert capture.grabs == 12

    def test_an_overshooting_seek_stops_rather_than_sampling_wrong_frames(self) -> None:
        # Landing past the target cannot be corrected by walking forward, and
        # reading there would bill a frame the sampler never selected. Stop and
        # bill only what was decoded.
        capture = _OvershootingCapture(40)
        frames = video_frames._sample(_FakeCv2, capture, total=40, budget=4)
        assert _source_indices(frames) == [0]

    def test_the_no_seek_walk_is_hard_capped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A broken-seek backend must not turn one item into an unbounded walk;
        # past the cap the item is refused with a typed 400 instead.
        monkeypatch.setattr(video_frames, "_MAX_SAMPLE_GRABS", 8)
        capture = _CountingCapture(400, seekable=False)
        with pytest.raises(VideoDecodeError, match="cannot seek"):
            video_frames._sample(_FakeCv2, capture, total=400, budget=4)
        assert capture.grabs <= 9

    def test_an_over_reported_length_bills_only_the_frames_decoded(self) -> None:
        # Containers can over-report CAP_PROP_FRAME_COUNT. Sampling then runs
        # off the end; the frames actually decoded are the ones the model gets
        # and therefore the only ones that may be billed.
        capture = _CountingCapture(12)
        frames = video_frames._sample(_FakeCv2, capture, total=40, budget=4)
        assert len(frames) == len(_source_indices(frames)) <= 4
        assert _source_indices(frames) == [0, 10]


class TestFailsClosed:
    """A video that cannot be decoded must fault, never silently bill zero."""

    def test_undecodable_bytes_raise_typed_invalid_input(self) -> None:
        with pytest.raises(VideoDecodeError):
            extract_frames(_video(b"not-a-video-at-all"))

    def test_decode_error_is_an_invalid_input_error(self) -> None:
        # Both ingress paths map InvalidInputError to INVALID_INPUT / HTTP 400.
        assert issubclass(VideoDecodeError, InvalidInputError)

    def test_empty_payload_raises(self) -> None:
        with pytest.raises(VideoDecodeError, match="no data"):
            extract_frames(_video(b""))

    def test_non_bytes_payload_raises_the_media_contract_error(self) -> None:
        with pytest.raises(InvalidMediaError):
            extract_frames({"data": "base64-string-that-was-never-decoded"})

    def test_non_positive_budget_raises(self, tmp_path: Path) -> None:
        with pytest.raises(VideoDecodeError, match="positive"):
            extract_frames(_video(_make_video(tmp_path, frames=2)), max_frames=0)

    def test_missing_decoder_raises_instead_of_returning_no_frames(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A GPU image with a broken OpenCV wheel must reject the request, never
        # return a billed success that processed zero frames. ``None`` in
        # sys.modules is exactly what a failed import leaves behind.
        monkeypatch.setitem(sys.modules, "cv2", None)
        with pytest.raises(VideoDecodeError, match="no video decoder"):
            extract_frames(_video(b"x" * 128))


class TestAdmissionCaps:
    def test_over_cap_bytes_are_rejected_before_decoding(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(video_frames, "MAX_VIDEO_BYTES", 8)
        with pytest.raises(VideoDecodeError, match="admission cap"):
            extract_frames(_video(b"x" * 9))

    def test_over_cap_duration_is_rejected(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # 30 frames at 10 fps = 3.0 s, over a 1 s cap.
        monkeypatch.setattr(video_frames, "MAX_VIDEO_DURATION_S", 1.0)
        with pytest.raises(VideoDecodeError, match="admission cap"):
            extract_frames(_video(_make_video(tmp_path, frames=30, fps=10.0)))

    def test_in_cap_duration_is_accepted(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(video_frames, "MAX_VIDEO_DURATION_S", 10.0)
        assert extract_frames(_video(_make_video(tmp_path, frames=30, fps=10.0)))

    def test_a_stream_exactly_at_the_scan_cap_is_accepted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The cap is INCLUSIVE: a stream of exactly `_MAX_SCANNED_FRAMES` is
        # within it. Reaching the cap is not the same as exceeding it, and
        # "tightening" the guard to `total >= cap` would reject admissible
        # input — you cannot know a stream is over the cap without grabbing
        # the frame past it.
        monkeypatch.setattr(video_frames, "_validated_frame_total", lambda *_: None)
        monkeypatch.setattr(video_frames, "_MAX_SCANNED_FRAMES", 6)
        frames = extract_frames(_video(_make_video(tmp_path, frames=6)), max_frames=3)
        assert _source_indices(frames) == [0, 2, 4]

    def test_scan_cap_bounds_metadata_less_containers(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Without fps metadata the duration cap degrades to a frame-scan cap;
        # both bound the same abuse (unbounded decode work for one item).
        monkeypatch.setattr(video_frames, "_validated_frame_total", lambda *_: None)
        monkeypatch.setattr(video_frames, "_MAX_SCANNED_FRAMES", 4)
        with pytest.raises(VideoDecodeError, match="scan cap"):
            extract_frames(_video(_make_video(tmp_path, frames=12)))


class TestBillingContractStaysInSyncWithTheGateway:
    """The managed reservation ceiling is a hand-mirrored copy of the budget.

    Settlement rejects a worker count above its reservation ceiling, so a
    Python budget raised past the gateway constant would turn every successful
    video encode into a billing fault. Nothing but this test binds the two.
    """

    def test_gateway_ceiling_covers_the_worker_budget(self) -> None:
        repo_root = Path(__file__).resolve().parents[4]
        dispatcher = repo_root / "packages" / "sie_cloud" / "gateway" / "src" / "dispatcher.rs"
        if not dispatcher.is_file():  # pragma: no cover - checkout without the gateway crate
            pytest.skip("gateway crate not present in this checkout")
        marker = "const VIDEO_MAX_SAMPLED_FRAMES: usize = "
        declarations = [line for line in dispatcher.read_text().splitlines() if marker in line]
        assert len(declarations) == 1, "expected exactly one gateway frame-budget constant"
        gateway_budget = int(declarations[0].split(marker)[1].split(";")[0].strip())
        assert gateway_budget >= MAX_SAMPLED_FRAMES
