"""Pure generation-stream timing helper for the worker telemetry facade.

OpenTelemetry instrument ownership and semantic event normalization live in
``worker_telemetry``.  This module only turns adapter stream lifecycle into
calls on that one facade; it creates no instruments and owns no exporter
state.
"""

from __future__ import annotations

import time

from sie_server.observability.worker_telemetry import worker_telemetry, worker_telemetry_enabled


class GenerationStreamTimer:
    """Turn one adapter stream into TTFT, TPOT, and token facade events."""

    __slots__ = (
        "_completion_yields",
        "_enabled",
        "_finalized",
        "_first_yield_at",
        "_grammar",
        "_last_yield_at",
        "_model",
        "_started_at",
    )

    def __init__(self, model: str, *, grammar: str = "none") -> None:
        self._model = model
        self._grammar = grammar
        self._enabled = worker_telemetry_enabled()
        self._started_at = time.perf_counter() if self._enabled else 0.0
        self._first_yield_at: float | None = None
        self._last_yield_at: float | None = None
        self._completion_yields = 0
        self._finalized = False

    def mark_yield(self, *, has_text: bool) -> None:
        if not self._enabled or not has_text:
            return
        now = time.perf_counter()
        if self._first_yield_at is None:
            self._first_yield_at = now
            worker_telemetry().first_token_observed(
                model=self._model,
                grammar=self._grammar,
                duration_s=now - self._started_at,
            )
        self._last_yield_at = now
        self._completion_yields += 1

    def finalize(
        self,
        *,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> None:
        if self._finalized:
            return
        self._finalized = True
        if not self._enabled:
            return
        tpot_s: float | None = None
        if (
            self._first_yield_at is not None
            and self._last_yield_at is not None
            and self._last_yield_at > self._first_yield_at
        ):
            denominator = completion_tokens if completion_tokens and completion_tokens > 0 else self._completion_yields
            if denominator > 0:
                window = self._last_yield_at - self._first_yield_at
                tpot_s = window / denominator
        worker_telemetry().stream_finished(
            model=self._model,
            grammar=self._grammar,
            tpot_s=tpot_s,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
