"""Process-local observations; filesystem reads occur only outside timed work."""

from __future__ import annotations

import gc
import math
import os
import resource
import time
from array import array
from pathlib import Path
from typing import Any

from tools.benchmarks.cross_encoder_lora.protocol import require

GC_FIELDS = ("collections", "collected", "uncollectable")

RUSAGE_FIELDS = ("ru_utime", "ru_stime", "ru_minflt", "ru_majflt", "ru_nvcsw", "ru_nivcsw")

OPTIONAL_FILES = ("cpu.stat", "cpu.stat.local", "cpu.pressure", "cpu.max", "cpu.weight", "cpuset.cpus.effective")

COUNTER_FILES = ("cpu.stat", "cpu.stat.local")

CONTROL_FILES = ("cpu.max", "cpu.weight", "cpuset.cpus.effective")

GC_GENERATIONS = 3

PSI_FIELDS = ("avg10", "avg60", "avg300", "total")

MAX_PERCENT = 100

MAX_CPU_WEIGHT = 10000


def unified_cgroup(proc: Path = Path("/proc/self")) -> dict[str, Any]:
    """Resolve the process's own namespace; never fall back to another cgroup."""
    try:
        rows = (proc / "cgroup").read_text().splitlines()
        membership = [row[3:] for row in rows if row.startswith("0::")]
        require(len(membership) == 1, "unified cgroup membership is unavailable or ambiguous")
        group = Path(membership[0])
        require(group.is_absolute() and ".." not in group.parts, "invalid cgroup membership")
        matches = []
        for row in (proc / "mountinfo").read_text().splitlines():
            left, right = row.split(" - ", 1)
            if right.split()[0] != "cgroup2":
                continue
            fields = left.split()
            root = Path(fields[3].replace("\\040", " ").replace("\\134", "\\"))
            mount = Path(fields[4].replace("\\040", " ").replace("\\134", "\\"))
            if group.is_relative_to(root):
                matches.append(mount / group.relative_to(root))
        require(len(matches) == 1, "cgroup mount cannot be resolved uniquely")
        path = matches[0]
        info = path.stat()
        return {"path": str(path), "device": info.st_dev, "inode": info.st_ino, "error": None}
    except (OSError, ValueError, IndexError, AssertionError) as error:
        return {"path": None, "error": f"{type(error).__name__}: {error}"}


def flat_counters(text: str) -> dict[str, int]:
    result = {}
    for row in text.splitlines():
        key, value = row.split()
        require(key not in result, "duplicate telemetry counter")
        result[key] = int(value)
        require(result[key] >= 0, "negative telemetry counter")
    require(bool(result), "empty exposed telemetry counter file")
    return result


def pressure(text: str) -> dict[str, dict[str, float | int]]:
    result = {}
    for row in text.splitlines():
        name, *parts = row.split()
        require(name in {"some", "full"} and name not in result, "invalid PSI row")
        values = dict(part.split("=", 1) for part in parts)
        require(len(parts) == len(PSI_FIELDS) and set(values) == set(PSI_FIELDS), "invalid PSI fields")
        parsed = {key: int(value) if key == "total" else float(value) for key, value in values.items()}
        require(all(math.isfinite(value) and value >= 0 for value in parsed.values()), "invalid PSI value")
        require(all(parsed[key] <= MAX_PERCENT for key in PSI_FIELDS[:-1]), "invalid PSI percentage")
        result[name] = parsed
    require("some" in result, "missing exposed PSI some row")
    return result


def validate_core(core: dict[str, Any]) -> None:
    require(
        all(isinstance(core[key], int) and core[key] > 0 for key in ("wall_ns", "monotonic_ns"))
        and core["gc_enabled"] is True,
        "invalid required clock/GC state",
    )
    require(
        set(core["rusage"]) == set(RUSAGE_FIELDS)
        and all(math.isfinite(value) and value >= 0 for value in core["rusage"].values()),
        "invalid required rusage",
    )
    for key in ("gc_thresholds", "gc_count"):
        require(
            len(core[key]) == GC_GENERATIONS and all(type(value) is int and value >= 0 for value in core[key]),
            "invalid required GC counts/thresholds",
        )
    require(core["gc_thresholds"][0] > 0, "automatic GC is disabled by its threshold")
    require(len(core["gc_stats"]) == GC_GENERATIONS, "missing required GC generations")
    require(
        all(
            set(row) == set(GC_FIELDS) and all(type(row[key]) is int and row[key] >= 0 for key in GC_FIELDS)
            for row in core["gc_stats"]
        ),
        "invalid required GC statistics",
    )


def control_value(name: str, text: str) -> str:
    if name == "cpu.max":
        quota, period = text.split()
        require((quota == "max" or int(quota) > 0) and int(period) > 0, "invalid exposed cpu.max")
    elif name == "cpu.weight":
        require(1 <= int(text) <= MAX_CPU_WEIGHT, "invalid exposed cpu.weight")
    else:
        for item in text.split(","):
            bounds = [int(value) for value in item.split("-")]
            require(len(bounds) in {1, 2} and bounds[0] >= 0 and bounds[-1] >= bounds[0], "invalid exposed cpuset")
    return text


def snapshot() -> dict[str, Any]:
    """Only called outside performance(), including all filesystem reads."""
    started = time.monotonic_ns()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    core = {
        "wall_ns": time.time_ns(),
        "monotonic_ns": started,
        "perf_counter_ns": time.perf_counter_ns(),
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "rusage": {key: getattr(usage, key) for key in RUSAGE_FIELDS},
        "gc_enabled": gc.isenabled(),
        "gc_thresholds": list(gc.get_threshold()),
        "gc_callback_ids": [id(callback) for callback in gc.callbacks],
        "gc_count": list(gc.get_count()),
        "gc_stats": gc.get_stats(),
    }
    validate_core(core)
    group = unified_cgroup()
    files: dict[str, Any] = {}
    errors = {}
    if group["path"] is not None:
        for name in OPTIONAL_FILES:
            try:
                text = (Path(group["path"]) / name).read_text().strip()
            except OSError as error:
                files[name] = None
                errors[name] = f"{type(error).__name__}: {error}"
                continue
            # Missing optional files are unavailability, but malformed exposed
            # counters must not be mistaken for trustworthy zero deltas.
            files[name] = (
                flat_counters(text)
                if name in COUNTER_FILES
                else pressure(text)
                if name == "cpu.pressure"
                else control_value(name, text)
            )
    else:
        files = dict.fromkeys(OPTIONAL_FILES)
        errors["cgroup"] = group["error"]
    affinity = None
    try:
        if hasattr(os, "sched_getaffinity"):
            affinity = sorted(os.sched_getaffinity(0))
            require(bool(affinity), "empty exposed affinity")
        else:
            errors["affinity"] = "sched_getaffinity unavailable on this platform"
    except OSError as error:
        errors["affinity"] = f"{type(error).__name__}: {error}"
    return {
        "qualification": False,
        "core": core,
        "cgroup": group,
        "cgroup_files": files,
        "optional_errors": errors,
        "affinity": affinity,
        "snapshot_read_ns": time.monotonic_ns() - started,
    }


def counter_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    require(set(before) == set(after), "counter fields changed within a window")
    result = {key: after[key] - value for key, value in before.items()}
    require(all(math.isfinite(value) and value >= 0 for value in result.values()), "telemetry counter reset")
    return result


def telemetry_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    first, last = before["core"], after["core"]
    validate_core(first)
    validate_core(last)
    require(
        last["wall_ns"] > first["wall_ns"] and last["monotonic_ns"] > first["monotonic_ns"],
        "required clock did not advance",
    )
    require(
        first["gc_enabled"] and last["gc_enabled"] and first["gc_thresholds"] == last["gc_thresholds"],
        "GC configuration changed",
    )
    group_before, group_after = before["cgroup"], after["cgroup"]
    require(
        all(group_before.get(key) == group_after.get(key) for key in ("path", "device", "inode")),
        "cgroup path or identity changed within a window",
    )
    require(before["affinity"] == after["affinity"], "process affinity changed within a window")
    for name in CONTROL_FILES:
        require(
            before["cgroup_files"].get(name) == after["cgroup_files"].get(name),
            "cgroup quota/weight/cpuset changed within a window",
        )
    optional = {}
    for name in COUNTER_FILES:
        a, b = before["cgroup_files"].get(name), after["cgroup_files"].get(name)
        optional[name] = counter_delta(a, b) if a is not None and b is not None else None
    a, b = before["cgroup_files"].get("cpu.pressure"), after["cgroup_files"].get("cpu.pressure")
    optional["cpu.pressure"] = (
        counter_delta({k: v["total"] for k, v in a.items()}, {k: v["total"] for k, v in b.items()})
        if a is not None and b is not None
        else None
    )
    return {
        "qualification": False,
        "scope": "Whole performance call, including sequential warmup, measurement, drain, validation and raw emission",
        "wall_s": (last["wall_ns"] - first["wall_ns"]) / 1e9,
        "monotonic_s": (last["monotonic_ns"] - first["monotonic_ns"]) / 1e9,
        "rusage": counter_delta(first["rusage"], last["rusage"]),
        "gc_stats": [counter_delta(a, b) for a, b in zip(first["gc_stats"], last["gc_stats"], strict=True)],
        "optional_cgroup": optional,
        "cpu_pressure_counters_available": all(optional.get(name) is not None for name in ("cpu.stat", "cpu.pressure")),
        "resource_telemetry_complete": before["affinity"] is not None
        and after["affinity"] is not None
        and all(
            before["cgroup_files"].get(name) is not None and after["cgroup_files"].get(name) is not None
            for name in OPTIONAL_FILES
        ),
        "optional_errors_before": before["optional_errors"],
        "optional_errors_after": after["optional_errors"],
    }


def _boundary_gc_projection(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    first, last = before["core"], after["core"]
    errors = []
    for core in (first, last):
        require(
            isinstance(core["gc_callback_ids"], list)
            and all(type(value) is int and value > 0 for value in core["gc_callback_ids"]),
            "invalid boundary GC callback identities",
        )
        require(
            len(core["gc_thresholds"]) == GC_GENERATIONS
            and all(type(value) is int and value >= 0 for value in core["gc_thresholds"]),
            "invalid boundary GC thresholds",
        )
    if first["pid"] != last["pid"]:
        errors.append("GC boundary process identity changed")
    if first["gc_callback_ids"] != last["gc_callback_ids"]:
        errors.append("pre-existing GC callback identities changed")
    if first["gc_enabled"] is not True or last["gc_enabled"] is not True:
        errors.append("ordinary automatic GC must remain enabled")
    if first["gc_thresholds"] != last["gc_thresholds"] or first["gc_thresholds"][0] <= 0:
        errors.append("GC thresholds changed or disabled automatic collection")
    return {
        "mode": "boundary-only",
        "pid": first["pid"],
        "scope": "Boundary state only; intervening changes and GC pauses are unobserved.",
        "integrity_passed": not errors,
        "errors": errors,
        "records": None,
        "intervals": None,
        "callback_self_ns": None,
        "estimated_observer_fraction": None,
        "cost_budget_passed": None,
        "unavailable_reason": "No per-event GC callbacks are installed by the qualification harness.",
    }


def boundary_gc_report(before: dict[str, Any], after: dict[str, Any], callbacks: tuple[Any, ...]) -> dict[str, Any]:
    """Check outside-clock state without installing callbacks or changing GC.

    Holding the original objects through both snapshots prevents identity reuse.
    This cannot observe an intervening change that is undone before the boundary.
    """
    result = _boundary_gc_projection(before, after)
    if (
        before["core"]["gc_callback_ids"] != [id(callback) for callback in callbacks]
        or len(gc.callbacks) != len(callbacks)
        or any(current is not prior for current, prior in zip(gc.callbacks, callbacks, strict=False))
    ):
        result["errors"].append("retained GC callback identity differs at a boundary")
        result["integrity_passed"] = False
    return result


def validate_boundary_gc(observation: dict[str, Any]) -> None:
    """Reject measured-event claims or contradictory raw boundary evidence."""
    expected = _boundary_gc_projection(observation["before"], observation["after"])
    require(
        expected["integrity_passed"] is True
        and observation["gc"]["integrity_passed"] is True
        and observation["gc"] == expected,
        "boundary-only GC state or unavailable observation fields differ",
    )


# Historical per-event diagnostic limits; qualification does not install this recorder.
MAX_CALLBACK_FRACTION = 0.001
CALLBACK_TAIL_ALLOWANCE_NS = 10_000
GC_RECORD_CAPACITY = 65_536
GC_RECORD_WIDTH = 6


class GCRecorder:
    """Bounded numeric records, with no I/O, locks or growing collections in callbacks.

    Python still creates transient scalar objects. The reported cost includes
    measured callback self time plus a fixed per-call tail allowance; it is an
    estimate with a conservative allowance, not proof of zero perturbation.
    """

    def __init__(self, capacity: int = GC_RECORD_CAPACITY) -> None:
        require(type(capacity) is int and capacity > 0, "invalid GC recorder capacity")
        self.capacity = capacity
        self.buffer = array("Q", [0]) * (capacity * GC_RECORD_WIDTH)
        self.count = 0
        self.overflow = False
        self.invalid = False
        self._hook: Any = None
        self._prior: tuple[Any, ...] = ()
        self._thresholds: tuple[int, ...] = ()
        self._closed = False
        self.lifecycle_errors: list[str] = []

    def _record(self, phase: str, info: dict[str, Any]) -> None:
        entered = time.perf_counter_ns()
        generation = info.get("generation")
        if phase not in ("start", "stop") or type(generation) is not int or not 0 <= generation < GC_GENERATIONS:
            self.invalid = True
            return
        if self.count >= self.capacity:
            self.overflow = True
            return
        collected, uncollectable = info.get("collected", 0), info.get("uncollectable", 0)
        if type(collected) is not int or type(uncollectable) is not int or collected < 0 or uncollectable < 0:
            self.invalid = True
            return
        offset = self.count * GC_RECORD_WIDTH
        self.buffer[offset] = entered
        self.buffer[offset + 1] = int(phase == "stop")
        self.buffer[offset + 2] = generation
        self.buffer[offset + 3] = collected
        self.buffer[offset + 4] = uncollectable
        self.count += 1
        self.buffer[offset + 5] = time.perf_counter_ns() - entered

    def start(self) -> None:
        require(self._hook is None and not self._closed, "GC recorder cannot be restarted")
        require(gc.isenabled() and gc.get_threshold()[0] > 0, "ordinary automatic GC must remain enabled")
        self._prior = tuple(gc.callbacks)
        self._thresholds = gc.get_threshold()
        self._hook = self._record
        gc.callbacks.append(self._hook)

    def close(self) -> None:
        if self._closed:
            return
        hook, self._hook = self._hook, None
        if hook is None:
            self.lifecycle_errors.append("GC recorder was never installed")
        else:
            matches = [index for index, item in enumerate(gc.callbacks) if item is hook]
            if len(matches) != 1:
                self.lifecycle_errors.append("owned GC callback identity changed")
            for index in reversed(matches):
                gc.callbacks.pop(index)
        if len(gc.callbacks) != len(self._prior) or any(
            current is not prior for current, prior in zip(gc.callbacks, self._prior, strict=False)
        ):
            self.lifecycle_errors.append("pre-existing GC callbacks changed")
        if not gc.isenabled() or gc.get_threshold() != self._thresholds:
            self.lifecycle_errors.append("GC enable state or thresholds changed")
        self._closed = True

    def report(self, start_s: float, end_s: float) -> dict[str, Any]:
        require(self._closed, "GC recorder must close before materializing evidence")
        require(math.isfinite(start_s) and math.isfinite(end_s) and 0 < start_s < end_s, "invalid GC measurement clock")
        begin, end = round(start_s * 1e9), round(end_s * 1e9)
        rows = [
            list(self.buffer[index * GC_RECORD_WIDTH : (index + 1) * GC_RECORD_WIDTH]) for index in range(self.count)
        ]
        pending: dict[int, int] = {}
        intervals = []
        pair_errors = []
        for stamp, stopped, generation, collected, uncollectable, _cost in rows:
            if not stopped:
                if generation in pending:
                    pair_errors.append("duplicate GC start")
                pending[generation] = stamp
            elif generation not in pending:
                pair_errors.append("GC stop has no observed start")
            else:
                first = pending.pop(generation)
                if stamp < first:
                    pair_errors.append("GC event clock reversed")
                intervals.append(
                    {
                        "generation": generation,
                        "start_ns": first,
                        "end_ns": stamp,
                        "collected": collected,
                        "uncollectable": uncollectable,
                        "overlap_ns": max(0, min(end, stamp) - max(begin, first)),
                    }
                )
        if pending:
            pair_errors.append("GC start has no observed stop")
        relevant = [row for row in rows if row[0] <= end and row[0] + row[5] + CALLBACK_TAIL_ALLOWANCE_NS >= begin]
        self_ns = sum(row[5] for row in relevant)
        allowance_ns = len(relevant) * CALLBACK_TAIL_ALLOWANCE_NS
        estimated_fraction = (self_ns + allowance_ns) / (end - begin)
        errors = [*self.lifecycle_errors, *pair_errors]
        if self.overflow:
            errors.append("GC recorder overflow")
        if self.invalid:
            errors.append("invalid GC callback event")
        integrity_passed = not errors
        cost_passed = estimated_fraction <= MAX_CALLBACK_FRACTION
        if not cost_passed:
            errors.append("GC observer cost exceeds prospective window budget")
        return {
            "pid": os.getpid(),
            "clock": "perf_counter_ns, verified against child/server handshake bounds",
            "columns": ["timestamp_ns", "stop", "generation", "collected", "uncollectable", "callback_self_ns"],
            "records": rows,
            "intervals": intervals,
            "window_start_ns": begin,
            "window_end_ns": end,
            "callback_count_intersecting_window": len(relevant),
            "callback_count_entered_in_window": sum(begin <= row[0] <= end for row in rows),
            "callback_self_ns": self_ns,
            "per_event_tail_allowance_ns": CALLBACK_TAIL_ALLOWANCE_NS,
            "estimated_observer_fraction": estimated_fraction,
            "maximum_observer_fraction": MAX_CALLBACK_FRACTION,
            "uncertainty": "Includes a fixed 10us/call tail allowance requiring a recorded preflight calibration; timing cost does not bound every scheduling/GC perturbation.",
            "integrity_passed": integrity_passed,
            "cost_budget_passed": cost_passed,
            "passed": not errors,
            "errors": errors,
        }


def bracket(before: dict[str, Any], after: dict[str, Any], clock: dict[str, float]) -> dict[str, Any]:
    """Keep enclosing GC/CPU/cgroup deltas distinct from request-level timing."""
    require(before["core"]["pid"] == after["core"]["pid"], "process identity changed")
    begin, end = clock["start"], clock["end"]
    require(
        before["core"]["perf_counter_ns"] / 1e9 <= begin < end <= after["core"]["perf_counter_ns"] / 1e9,
        "process telemetry does not bracket the measured monotonic interval",
    )
    delta = telemetry_delta(before, after)
    delta["scope"] = (
        "Enclosing process interval around the load-generator timer; includes boundary handshake/observation overhead, not exact per-request CPU time"
    )
    delta["cgroup_scope"] = "Shared container counters; never sum client and server observations"
    return {"before": before, "after": after, "delta": delta}
