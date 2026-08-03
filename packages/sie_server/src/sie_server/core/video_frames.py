"""Video frame extraction and admission caps for the frames-as-images meter.

Sampled video frames are the model's real visual input, so they bill as §7
``images`` units on the count of frames the worker **actually processed** —
never a request-declared or estimated number. This module is the single seam
that turns wire ``VideoInput`` bytes into PIL frames, so exactly one place owns
the sampling budget that both the biller and the managed reservation ceiling
depend on.

Three invariants hold here:

- **Bounded.** :data:`MAX_SAMPLED_FRAMES` caps the frames one item can ever
  yield. The managed gateway reserves ``images`` for a video item from that
  same constant (``VIDEO_MAX_SAMPLED_FRAMES`` in
  ``packages/sie_cloud/gateway/src/dispatcher.rs``), and settlement rejects a
  worker count above its reservation ceiling — so this budget is a billing
  contract, deliberately NOT environment-tunable.
- **Fail-closed.** Every decode failure raises :class:`VideoDecodeError`
  (an :class:`~sie_server.types.inputs.InvalidInputError`, surfaced as
  ``INVALID_INPUT`` / HTTP 400 on both the HTTP and queue paths). A missing or
  broken decoder in a GPU image can therefore never masquerade as a successful
  zero-frame encode that silently drops the customer's video.
- **Deterministic.** Sampling is evenly spaced over the stream and fills the
  budget exactly, so the same bytes always yield the same frames and therefore
  the same bill.

Decode work is bounded by the frame budget, not by the stream length: sampling
seeks to the target indices and decodes only those. That matters beyond
latency — on the batched worker path ``adapter.encode`` runs on
``ModelWorker._inference_executor``, a single-threaded pool that serializes the
model's GPU work, so walking a long stream there would stall every
co-scheduled request. The residual costs are one seek per sampled frame
(FFmpeg decodes from the preceding keyframe, so at most one GOP each) and, for
containers that report no usable metadata, the one grab-only counting pass
under :data:`_MAX_SCANNED_FRAMES`.

TODO(#2433): move extraction off the inference executor entirely — decode in
per-item preprocessing (``EncodePipeline._prepare_batch``) and hand the frames
to the adapter through ``PreparedItem.payload``, so the GPU thread only ever
sees decoded frames. That needs a prepared-item payload the video-capable
adapters actually read, which today they do not (``get_preprocessor`` returns
``None``), so it is a follow-up rather than part of this seam.
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
from collections.abc import Callable
from typing import Any, Final

from PIL import Image

from sie_server.types.inputs import InvalidInputError, media_bytes

logger = logging.getLogger(__name__)

# Billing contract — see the module docstring. Raising this without raising the
# gateway's `VIDEO_MAX_SAMPLED_FRAMES` would let a settled frame count exceed
# its reservation ceiling and turn a successful encode into a billing fault, so
# it is a constant rather than an environment knob.
MAX_SAMPLED_FRAMES: Final[int] = 32


# Admission caps: abuse rails bounding decode work per item, not model limits.
# Generous by design — a minute of 1080p H.264 sits far below the byte cap.
#
# Unlike MAX_SAMPLED_FRAMES these ARE env knobs, but only LOWERING the byte cap
# is meaningful behind the managed gateway: it mirrors this default as the
# compile-time `METERED_VIDEO_INPUT_BYTE_CAP` and rejects an over-cap payload
# pre-reservation, so raising SIE_MAX_VIDEO_BYTES past 256 MiB changes nothing
# there and only widens what self-hosted deployments accept. The duration cap
# has no gateway mirror at all (the gateway holds no decoder), so it is
# authoritative on every path.
#
# Parsed defensively: these run at IMPORT time, so a typo in either knob would
# raise out of `import sie_server` and take down startup for the whole server —
# including every text-only path that never touches video. A malformed value
# falls back to the default and logs, which keeps the blast radius at the knob.
def _bounded_env[T: (int, float)](name: str, default: str, cast: Callable[[str], T]) -> T:
    raw = os.environ.get(name, default)
    try:
        return cast(raw)
    except (TypeError, ValueError):
        logger.warning(
            "%s=%r is not a valid number; falling back to the default %s",
            name,
            raw,
            default,
        )
        return cast(default)


MAX_VIDEO_BYTES: Final[int] = _bounded_env("SIE_MAX_VIDEO_BYTES", str(256 * 1024 * 1024), int)
MAX_VIDEO_DURATION_S: Final[float] = _bounded_env("SIE_MAX_VIDEO_DURATION_S", "1800", float)

# Containers without usable fps/frame-count metadata cannot be duration-checked
# before decoding, so the duration cap degrades to this scan cap on decoded
# frames (1800 s at 60 fps). Both are ceilings on the same abuse: unbounded
# decode work for one item.
_MAX_SCANNED_FRAMES: Final[int] = 108_000

# Hard cap on ``grab()`` calls for the no-seek sampling fallback. Sampling is
# seek-based: it decodes ONLY the target frames, so its cost is the frame budget
# and not the stream length. A backend that cannot position the capture has to
# walk forward instead, and that walk is bounded here rather than allowed to
# scale with the stream — a decode holds a worker thread, and on the batched
# path that thread is the model's single GPU-serialization executor, so an
# unbounded walk stalls every co-scheduled request. Beyond the cap the item is
# refused with a typed 400 instead of monopolising the worker. Sized for a
# 2-minute 30 fps clip; only a backend with broken seeking ever reaches it.
_MAX_SAMPLE_GRABS: Final[int] = 3_600

_FORMAT_SUFFIXES: Final[dict[str, str]] = {
    "mp4": ".mp4",
    "m4v": ".mp4",
    "mov": ".mov",
    "webm": ".webm",
    "mkv": ".mkv",
    "avi": ".avi",
    "gif": ".gif",
}


class VideoDecodeError(InvalidInputError):
    """A video input could not be decoded, or violates an admission cap.

    Subclasses :class:`~sie_server.types.inputs.InvalidInputError` so both
    ingress paths surface it as structured ``INVALID_INPUT`` (HTTP 400): a
    caller-controlled input that the worker refuses is a client error, and —
    critically — never a billed success that quietly processed zero frames.
    """


def _suffix_for(video: Any) -> str:
    """Map the wire ``format`` hint to a temp-file suffix for the decoder.

    The hint is advisory: FFmpeg sniffs the container, and the suffix only
    helps it pick a demuxer faster. An unknown hint falls back to ``.mp4``.
    """
    hint = video.get("format") if isinstance(video, dict) else None
    if isinstance(hint, str):
        return _FORMAT_SUFFIXES.get(hint.strip().lower().lstrip("."), ".mp4")
    return ".mp4"


def _bgr_to_pil(frame: Any) -> Image.Image:
    """Convert one decoded OpenCV BGR frame to an RGB PIL image."""
    return Image.fromarray(frame[:, :, ::-1])


def extract_frames(video: Any, *, max_frames: int = MAX_SAMPLED_FRAMES) -> list[Image.Image]:
    """Decode ``video`` and return up to ``max_frames`` uniformly sampled frames.

    The returned list length is the item's authoritative billable frame count:
    every frame here was handed to the model, and no frame reaches the model
    without appearing here.

    Args:
        video: A wire :class:`~sie_server.types.inputs.VideoInput` mapping.
        max_frames: Sampling budget; must not exceed :data:`MAX_SAMPLED_FRAMES`.

    Returns:
        The sampled frames as RGB PIL images, in stream order.

    Raises:
        VideoDecodeError: The bytes are empty, exceed an admission cap, or no
            decoder could open/read them.
        InvalidMediaError: ``video`` violates the wire media contract.
    """
    budget = min(max_frames, MAX_SAMPLED_FRAMES)
    if budget < 1:
        msg = f"video frame budget must be positive, got {max_frames}"
        raise VideoDecodeError(msg)

    data = media_bytes(video, kind="video")
    if not data:
        raise VideoDecodeError("video input carries no data")
    if len(data) > MAX_VIDEO_BYTES:
        msg = f"video input is {len(data)} bytes, exceeding the {MAX_VIDEO_BYTES}-byte admission cap"
        raise VideoDecodeError(msg)

    cv2 = _load_decoder()
    # VideoCapture reads from a path, not memory. Delete-on-close is deferred to
    # the context manager so the decoder can reopen the file by name on FFmpeg
    # backends that re-open for seeking.
    with tempfile.NamedTemporaryFile(suffix=_suffix_for(video)) as handle:
        handle.write(data)
        handle.flush()
        capture = cv2.VideoCapture(handle.name)
        try:
            if not capture.isOpened():
                raise VideoDecodeError("video input could not be opened by the decoder")
            total = _validated_frame_total(cv2, capture)
            return _sample(cv2, capture, total=total, budget=budget)
        finally:
            capture.release()


def _load_decoder() -> Any:
    """Import the OpenCV decoder, failing closed when the image lacks it.

    cv2 is a declared ``sie_server`` dependency, but decoder wheels are known to
    be broken in some GPU images (issue #2433). Importing at point of use keeps
    a broken image from taking down server startup, and converts the breakage
    into a typed 4xx rather than a silent zero-frame success.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency present in-tree
        msg = f"no video decoder is available in this image: {exc}"
        raise VideoDecodeError(msg) from exc
    return cv2


def _validated_frame_total(cv2: Any, capture: Any) -> int | None:
    """Return the container's frame count, enforcing the duration cap.

    ``None`` means the container reported unusable metadata; the caller then
    counts frames itself under :data:`_MAX_SCANNED_FRAMES`.

    Both properties are C doubles, so a container with a broken header can
    report NaN or infinity. Those are truthy, so ``or 0`` does not filter them
    and ``int()`` would raise ``ValueError`` / ``OverflowError`` — escaping this
    module's typed-failure contract as a generic 500. They are checked for
    finiteness BEFORE any conversion and treated as unusable metadata.
    """
    raw_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if not math.isfinite(raw_count) or raw_count <= 0 or not math.isfinite(fps) or fps <= 0.0:
        return None
    frame_count = int(raw_count)
    duration_s = frame_count / fps
    if duration_s > MAX_VIDEO_DURATION_S:
        msg = f"video input is {duration_s:.1f}s long, exceeding the {MAX_VIDEO_DURATION_S}s admission cap"
        raise VideoDecodeError(msg)
    return frame_count


def _count_frames(capture: Any) -> int:
    """Count decodable frames with grab-only scanning (no pixel decode)."""
    total = 0
    while capture.grab():
        total += 1
        if total > _MAX_SCANNED_FRAMES:
            msg = (
                f"video input exceeds the {_MAX_SCANNED_FRAMES}-frame scan cap "
                "(container reported no usable duration metadata)"
            )
            raise VideoDecodeError(msg)
    return total


def _target_indices(total: int, budget: int) -> list[int]:
    """Evenly spaced source indices for one item's sampling budget.

    Returns exactly ``min(total, budget)`` strictly increasing indices spread
    over ``[0, total)``. A uniform *stride* (``ceil(total / budget)``) silently
    under-fills the budget for nearly every length — 33 frames at budget 32
    yields 17 — which halves both the visual context the model sees and the
    customer's bill at a boundary invisible to them. Even spacing fills the
    budget for every ``total >= budget`` and stays deterministic, so the same
    bytes keep producing the same frames and therefore the same bill.
    """
    count = min(total, budget)
    return [(index * total) // count for index in range(count)]


def _sample(cv2: Any, capture: Any, *, total: int | None, budget: int) -> list[Image.Image]:
    """Sample exactly ``min(total, budget)`` evenly spaced frames.

    Decoding is seek-based (see :func:`_decode_targets`): only the target
    frames are decoded, so the cost is the frame budget rather than the stream
    length.
    """
    if total is None:
        total = _count_frames(capture)
        # Rewind for the decode pass. A backend that cannot seek leaves the
        # capture exhausted, which the empty-result guard below catches.
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if total <= 0:
        raise VideoDecodeError("video input contains no decodable frames")

    frames = _decode_targets(cv2, capture, _target_indices(total, budget))
    if not frames:
        raise VideoDecodeError("video input contains no decodable frames")
    return frames


def _seek_to(cv2: Any, capture: Any, index: int) -> int | None:
    """Ask the backend to position the capture at ``index``.

    Returns where the capture actually LANDED, which is not always ``index``:
    a backend may snap backwards to the preceding keyframe, and one that
    silently ignores the request keeps reporting its old position. The caller
    must therefore resynchronise its logical cursor from the return value —
    assuming the landing equals the request would decode the wrong frames.

    ``None`` means the backend gave no usable position — ``set`` refused, or
    the reported position is missing or non-finite (the property is a C double,
    and ``int(nan)`` would raise mid-decode, escaping this module's typed
    contract). The caller then keeps its existing cursor and walks the gap.
    """
    if not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
        return None
    position = capture.get(cv2.CAP_PROP_POS_FRAMES)
    if position is None or not math.isfinite(float(position)):
        return None
    return int(position)


def _decode_targets(cv2: Any, capture: Any, targets: list[int]) -> list[Image.Image]:
    """Decode exactly the frames at ``targets`` (strictly increasing).

    Each gap between targets is crossed by seeking, so a normal stream costs
    one seek and one pixel decode per sampled frame regardless of its length.

    The cursor is always resynchronised from where the capture actually landed
    (see :func:`_seek_to`), never from where the seek was aimed. A backend that
    snaps backwards to a keyframe then pays a short forward ``grab()`` walk to
    reach the target; one that refuses to seek at all walks the whole gap,
    stickily, without re-asking. Either way the walk is bounded in total by
    :data:`_MAX_SAMPLE_GRABS`, so no backend can turn one item into an
    unbounded walk on the worker's inference thread.

    A read that fails part-way through is a container that over-reported its
    length (or a truncated tail): the frames already decoded are returned,
    because those are the frames the model gets and therefore exactly what may
    be billed.
    """
    frames: list[Image.Image] = []
    next_index = 0  # source index the capture will return on the next read
    seekable = True
    grabs = 0
    for target in targets:
        if target != next_index and seekable:
            landing = _seek_to(cv2, capture, target)
            if landing is None:
                # Refused: the capture did not move, so the cursor still holds
                # and the walk below crosses the gap. Do not ask again.
                seekable = False
            elif landing > target:
                # Overshot the target. Walking forward cannot come back, and
                # reading here would silently sample the wrong frame, so stop
                # and bill only the frames already decoded.
                logger.warning(
                    "Video sampling stopped early: seek to frame %d landed at %d (overshoot); "
                    "billing the %d frame(s) decoded so far out of %d requested",
                    target,
                    landing,
                    len(frames),
                    len(targets),
                )
                return frames
            else:
                # Exact, or snapped back to a keyframe: adopt the real position
                # and let the bounded walk cover whatever remains.
                next_index = landing
        while next_index < target:
            grabs += 1
            if grabs > _MAX_SAMPLE_GRABS:
                msg = (
                    f"video input needs more than {_MAX_SAMPLE_GRABS} skipped frames to sample "
                    "(this decoder backend cannot seek accurately)"
                )
                raise VideoDecodeError(msg)
            if not capture.grab():  # skip without paying the pixel-decode cost
                logger.warning(
                    "Video sampling stopped early: stream ended while skipping to frame %d; "
                    "billing the %d frame(s) decoded so far out of %d requested",
                    target,
                    len(frames),
                    len(targets),
                )
                return frames
            next_index += 1
        ok, frame = capture.read()
        if not ok:
            logger.warning(
                "Video sampling stopped early: frame %d could not be read (container over-reported "
                "its length, or the tail is truncated); billing the %d frame(s) decoded so far "
                "out of %d requested",
                target,
                len(frames),
                len(targets),
            )
            break
        frames.append(_bgr_to_pil(frame))
        next_index = target + 1
    return frames
