"""Request timing utilities.

Tracks latency breakdown for encode requests per DESIGN.md Section 5.7.

Timing headers:
- X-Queue-Time: Time waiting in queue (ms)
- X-Tokenization-Time: Time tokenizing (ms)
- X-Inference-Time: GPU forward pass (ms)
- X-Postprocessing-Time: Postprocessor transforms (ms), only if > 0
- X-Total-Time: End-to-end latency (ms)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class RequestTiming:
    """Timing information for a single request.

    All times are in milliseconds. Fields are set as each stage completes.
    """

    # Timestamps (monotonic, in seconds)
    _start_time: float = field(default_factory=time.monotonic)
    _tokenize_start: float | None = field(default=None, repr=False)
    _tokenize_end: float | None = field(default=None, repr=False)
    _queue_start: float | None = field(default=None, repr=False)
    _inference_start: float | None = field(default=None, repr=False)
    _inference_end: float | None = field(default=None, repr=False)
    _postprocess_start: float | None = field(default=None, repr=False)
    _postprocess_end: float | None = field(default=None, repr=False)
    _end_time: float | None = field(default=None, repr=False)

    def start_tokenization(self) -> None:
        """Mark tokenization start."""
        self._tokenize_start = time.monotonic()

    def end_tokenization(self) -> None:
        """Mark tokenization end."""
        self._tokenize_end = time.monotonic()

    def start_queue(self) -> None:
        """Mark queue start (request submitted to worker)."""
        self._queue_start = time.monotonic()

    def start_inference(self) -> None:
        """Mark inference start (batch picked up by worker)."""
        self._inference_start = time.monotonic()

    def end_inference(self) -> None:
        """Mark inference end."""
        self._inference_end = time.monotonic()

    def start_postprocessing(self) -> None:
        """Mark postprocessing start."""
        self._postprocess_start = time.monotonic()

    def end_postprocessing(self) -> None:
        """Mark postprocessing end."""
        self._postprocess_end = time.monotonic()

    def add_postprocessing_ms(self, elapsed_ms: float) -> None:
        """Add postprocessing time directly (for batch-level postprocessing).

        Use this when postprocessing time is tracked externally (e.g., by
        PostprocessorRegistry) and needs to be recorded on the request timing.

        Args:
            elapsed_ms: Postprocessing time in milliseconds.
        """
        # Create synthetic timestamps from elapsed time
        if elapsed_ms > 0:
            now = time.monotonic()
            self._postprocess_start = now - (elapsed_ms / 1000)
            self._postprocess_end = now

    def finish(self) -> None:
        """Mark request complete."""
        self._end_time = time.monotonic()

    @property
    def tokenization_ms(self) -> float:
        """Return tokenization time in milliseconds."""
        if self._tokenize_start is None or self._tokenize_end is None:
            return 0.0
        return (self._tokenize_end - self._tokenize_start) * 1000

    @property
    def queue_ms(self) -> float:
        """Return queue wait time in milliseconds."""
        if self._queue_start is None or self._inference_start is None:
            return 0.0
        return (self._inference_start - self._queue_start) * 1000

    @property
    def inference_ms(self) -> float:
        """Return inference time in milliseconds."""
        if self._inference_start is None or self._inference_end is None:
            return 0.0
        return (self._inference_end - self._inference_start) * 1000

    @property
    def postprocessing_ms(self) -> float:
        """Return postprocessing time in milliseconds."""
        if self._postprocess_start is None or self._postprocess_end is None:
            return 0.0
        return (self._postprocess_end - self._postprocess_start) * 1000

    @property
    def total_ms(self) -> float:
        """Return total end-to-end time in milliseconds."""
        end = self._end_time or time.monotonic()
        return (end - self._start_time) * 1000

    def to_headers(self) -> dict[str, str]:
        """Convert timing to HTTP response headers.

        Returns:
            Dict of header name to value (time in ms, formatted as string).
        """
        headers = {
            "X-Queue-Time": f"{self.queue_ms:.2f}",
            "X-Tokenization-Time": f"{self.tokenization_ms:.2f}",
            "X-Inference-Time": f"{self.inference_ms:.2f}",
            "X-Total-Time": f"{self.total_ms:.2f}",
        }
        # Only include postprocessing if it was recorded
        if self.postprocessing_ms > 0:
            headers["X-Postprocessing-Time"] = f"{self.postprocessing_ms:.2f}"
        return headers
