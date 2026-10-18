"""GC recording is bounded and diagnostic counters keep their actual scope."""

from __future__ import annotations

import copy
import gc
import time
from array import array
from pathlib import Path
from typing import Any

import pytest

from tools.benchmarks.cross_encoder_lora import process_metrics as metrics


def test_boundary_observation_preserves_gc_without_measuring_events() -> None:
    callbacks, thresholds = tuple(gc.callbacks), gc.get_threshold()
    before = metrics.snapshot()
    after = metrics.snapshot()
    value = metrics.boundary_gc_report(before, after, callbacks)
    metrics.validate_boundary_gc({"before": before, "after": after, "gc": value})
    assert value["mode"] == "boundary-only"
    assert value["integrity_passed"]
    assert all(
        value[key] is None
        for key in ("records", "intervals", "callback_self_ns", "estimated_observer_fraction", "cost_budget_passed")
    )
    assert "unobserved" in value["scope"]
    assert before["core"]["gc_callback_ids"] == after["core"]["gc_callback_ids"] == [id(item) for item in callbacks]
    assert len(gc.callbacks) == len(callbacks)
    assert all(current is prior for current, prior in zip(gc.callbacks, callbacks, strict=True))
    assert gc.isenabled()
    assert gc.get_threshold() == thresholds


def test_boundary_callback_identity_check_does_not_use_callable_equality() -> None:
    class Permissive:
        def __call__(self, _phase: str, _info: dict[str, Any]) -> None:
            pass

        def __eq__(self, _other: object) -> bool:
            return True

        def __hash__(self) -> int:
            return 0

    original, replacement = Permissive(), Permissive()
    previous = tuple(gc.callbacks)
    gc.callbacks.append(original)
    callbacks = tuple(gc.callbacks)
    try:
        before = metrics.snapshot()
        gc.callbacks[-1] = replacement
        after = metrics.snapshot()
        value = metrics.boundary_gc_report(before, after, callbacks)
        assert not value["integrity_passed"]
        with pytest.raises(AssertionError, match="boundary-only"):
            metrics.validate_boundary_gc({"before": before, "after": after, "gc": value})
        # Even forged matching snapshot IDs cannot hide a changed live object
        # from the local retained-reference check.
        after["core"]["gc_callback_ids"] = before["core"]["gc_callback_ids"]
        assert not metrics.boundary_gc_report(before, after, callbacks)["integrity_passed"]
    finally:
        gc.callbacks[:] = previous


def test_boundary_threshold_mutation_and_disabled_gc_fail_without_restoring_them_silently() -> None:
    thresholds = gc.get_threshold()
    callbacks = tuple(gc.callbacks)
    before = metrics.snapshot()
    try:
        gc.set_threshold(thresholds[0] + 1, *thresholds[1:])
        after = metrics.snapshot()
        value = metrics.boundary_gc_report(before, after, callbacks)
        assert not value["integrity_passed"]
        assert gc.get_threshold()[0] == thresholds[0] + 1
        gc.disable()
        with pytest.raises(AssertionError, match="GC state"):
            metrics.snapshot()
        after = copy.deepcopy(before)
        after["core"]["gc_enabled"] = False
        assert not metrics.boundary_gc_report(before, after, callbacks)["integrity_passed"]
        assert not gc.isenabled()
    finally:
        gc.set_threshold(*thresholds)
        gc.enable()


def report(rows: list[list[int]], start: float = 1.0, end: float = 2.0) -> dict[str, Any]:
    recorder = metrics.GCRecorder(capacity=max(1, len(rows)))
    recorder.start()
    recorder.close()
    recorder.count = len(rows)
    recorder.buffer = array("Q", [value for row in rows for value in row])
    return recorder.report(start, end)


def test_actual_gc_records_preserve_callbacks_thresholds_and_pairing() -> None:
    callbacks = tuple(gc.callbacks)
    thresholds = gc.get_threshold()
    recorder = metrics.GCRecorder()
    before = metrics.snapshot()
    recorder.start()
    started = time.perf_counter()
    cycle = []
    cycle.append(cycle)
    del cycle
    gc.collect()
    ended = time.perf_counter()
    recorder.close()
    after = metrics.snapshot()
    value = recorder.report(started, ended)
    assert value["integrity_passed"]
    assert value["records"]
    assert any(interval["generation"] == 2 and interval["collected"] >= 1 for interval in value["intervals"])
    assert tuple(gc.callbacks) == callbacks
    assert gc.get_threshold() == thresholds
    assert gc.isenabled()
    assert recorder._hook is None
    bracket = metrics.bracket(before, after, {"start": started, "end": ended})
    assert "Enclosing process interval" in bracket["delta"]["scope"]
    assert "never sum" in bracket["delta"]["cgroup_scope"]


def test_callback_straddling_start_is_charged_and_collection_overlap_reported() -> None:
    value = report([[999_999_000, 0, 0, 0, 0, 500], [1_100_000_000, 1, 0, 4, 0, 500]])
    assert value["passed"]
    assert value["callback_count_entered_in_window"] == 1
    assert value["callback_count_intersecting_window"] == 2
    assert value["callback_self_ns"] == 1000
    assert value["intervals"][0]["overlap_ns"] == 100_000_000
    assert value["estimated_observer_fraction"] == 0.000021


@pytest.mark.parametrize(
    "rows",
    [
        [[1_100_000_000, 0, 0, 0, 0, 1]],
        [[1_100_000_000, 1, 0, 0, 0, 1]],
        [[1_200_000_000, 0, 0, 0, 0, 1], [1_100_000_000, 1, 0, 0, 0, 1]],
        [[1_100_000_000, 0, 0, 0, 0, 1], [1_200_000_000, 0, 0, 0, 0, 1]],
    ],
)
def test_incomplete_or_reversed_gc_events_fail_integrity(rows: list[list[int]]) -> None:
    assert not report(rows)["integrity_passed"]


def test_overflow_and_invalid_events_fail_without_unbounded_growth() -> None:
    recorder = metrics.GCRecorder(capacity=1)
    recorder.start()
    recorder._record("start", {"generation": 0})
    recorder._record("stop", {"generation": 0})
    recorder._record("bad", {"generation": 0})
    recorder.close()
    value = recorder.report(1.0, time.perf_counter())
    assert len(recorder.buffer) == metrics.GC_RECORD_WIDTH
    assert not value["integrity_passed"]
    assert "GC recorder overflow" in value["errors"]
    assert "invalid GC callback event" in value["errors"]


def test_prior_callbacks_are_preserved_and_foreign_change_rejects() -> None:
    def prior(_phase: str, _info: dict[str, Any]) -> None:
        pass

    def unexpected(_phase: str, _info: dict[str, Any]) -> None:
        pass

    gc.callbacks.append(prior)
    recorder = metrics.GCRecorder()
    try:
        recorder.start()
        gc.callbacks.append(unexpected)
        recorder.close()
        assert prior in gc.callbacks
        assert unexpected in gc.callbacks
        assert "pre-existing GC callbacks changed" in recorder.report(1.0, 2.0)["errors"]
    finally:
        recorder.close()
        gc.callbacks.remove(prior)
        gc.callbacks.remove(unexpected)


def test_cost_limit_never_changes_performance_gates() -> None:
    value = report([[1_100_000_000, 0, 0, 0, 0, 600_000], [1_200_000_000, 1, 0, 0, 0, 600_000]])
    assert value["integrity_passed"]
    assert not value["cost_budget_passed"]
    assert not value["passed"]
    assert value["maximum_observer_fraction"] == 0.001


@pytest.mark.parametrize(("started", "ended"), [(2.0, 1.0), (1.0, 1.0), (float("nan"), 2.0)])
def test_invalid_measurement_clock_rejected(started: float, ended: float) -> None:
    with pytest.raises(AssertionError, match="clock"):
        report([], started, ended)


def test_callable_equality_cannot_hide_changed_callback_identity() -> None:
    class Permissive:
        def __call__(self, _phase: str, _info: dict[str, Any]) -> None:
            pass

        def __eq__(self, _other: object) -> bool:
            return True

        def __hash__(self) -> int:
            return 0

    original, replacement = Permissive(), Permissive()
    gc.callbacks.append(original)
    recorder = metrics.GCRecorder()
    try:
        recorder.start()
        index = next(index for index, value in enumerate(gc.callbacks) if value is original)
        gc.callbacks[index] = replacement
        recorder.close()
        assert "pre-existing GC callbacks changed" in recorder.report(1.0, 2.0)["errors"]
    finally:
        recorder.close()
        gc.callbacks[:] = [value for value in gc.callbacks if value is not original and value is not replacement]


def test_calibration_uses_six_units_fixed_allowance_and_signed_residuals() -> None:
    from tools.benchmarks.cross_encoder_lora.calibrate_gc import bound

    assert bound([-100.0] * 6)["mean_ns"] == -100.0
    assert bound([10_000.0] * 6)["passed"]
    assert not bound([10_000.1] * 6)["passed"]
    uncertain = bound([0, 0, 0, 10_000, 10_000, 10_000])
    assert uncertain["one_sided_95_upper_ns"] > uncertain["mean_ns"]
    with pytest.raises(AssertionError, match="six"):
        bound([1.0] * 5)


def test_calibration_preserves_gc_and_exercises_nonzero_fields() -> None:
    from tools.benchmarks.cross_encoder_lora.calibrate_gc import collect_arm, direct

    prior = tuple(gc.callbacks)
    thresholds = gc.get_threshold()
    for observed in (False, True):
        value = collect_arm(observed, 5)
        assert len(value["duration_ns"]) == 5
        assert len(value["records"]) == (10 if observed else 0)
    for generation in range(3):
        value = direct(generation, 5)
        assert value["info"]["collected"] > 0
        assert value["info"]["uncollectable"] > 0
        assert len(value["residual_ns"]) == 10
    assert tuple(gc.callbacks) == prior
    assert gc.get_threshold() == thresholds
    assert gc.isenabled()


def test_calibration_failure_writes_partial_raw_arms_before_reraising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    import sys

    from tools.benchmarks.cross_encoder_lora import calibrate_gc

    class ExtraStop(metrics.GCRecorder):
        def _record(self, phase: str, info: dict[str, Any]) -> None:
            super()._record(phase, info)
            if phase == "stop":
                super()._record(phase, info)

    monkeypatch.setattr(calibrate_gc, "GCRecorder", ExtraStop)
    monkeypatch.setattr(calibrate_gc, "COLLECTIONS", 3)
    output = tmp_path / "failed-calibration.json"
    monkeypatch.setattr(sys, "argv", ["calibrate_gc", "--output", str(output)])
    with pytest.raises(AssertionError, match="extra/missing"):
        calibrate_gc.main()
    value = json.loads(output.read_text())
    assert value["passed"] is False
    assert value["qualification"] is False
    assert value["error"]["type"] == "AssertionError"
    assert value["blocks"][0]["arms"]["False"]["passed"]
    observed = value["blocks"][0]["arms"]["True"]
    assert not observed["passed"]
    assert observed["completed"] == 3
    assert len(observed["duration_ns"]) == 3
    assert len(observed["records"]) == 9
