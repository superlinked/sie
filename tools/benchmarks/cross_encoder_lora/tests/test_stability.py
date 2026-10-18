"""Fixed baseline diagnostic controls; CPU smoke never qualifies GPU performance."""

# Owned Python children run the exact local diagnostic, without a shell.
# ruff: noqa: S603

from __future__ import annotations

import ast
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tools.benchmarks.cross_encoder_lora import prepare as source_preparer
from tools.benchmarks.cross_encoder_lora import process_metrics, worker
from tools.benchmarks.cross_encoder_lora import stability as probe
from tools.benchmarks.cross_encoder_lora.evidence import digest, fingerprint
from tools.benchmarks.cross_encoder_lora.fixtures import prepare
from tools.benchmarks.cross_encoder_lora.protocol import BASE_REVISION
from tools.benchmarks.cross_encoder_lora.runner import ROOT, extract, source_environment
from tools.benchmarks.cross_encoder_lora.tests.test_probe import tiny_snapshot


def observed(tick: int = 1) -> dict[str, Any]:
    return {
        "qualification": False,
        "core": {
            "wall_ns": tick * 1000000000,
            "monotonic_ns": tick * 1000000000,
            "rusage": dict.fromkeys(process_metrics.RUSAGE_FIELDS, tick),
            "gc_enabled": True,
            "gc_thresholds": [700, 10, 10],
            "gc_count": [tick, 0, 0],
            "gc_stats": [dict.fromkeys(process_metrics.GC_FIELDS, tick) for _ in range(3)],
        },
        "cgroup": {"path": None, "error": "unavailable"},
        "cgroup_files": dict.fromkeys(process_metrics.OPTIONAL_FILES),
        "optional_errors": {"cgroup": "unavailable"},
        "affinity": None,
        "snapshot_read_ns": 1,
    }


def process_rows() -> list[dict[str, Any]]:
    return [
        {
            "qualification": False,
            "completed": True,
            "shutdown_complete": True,
            "process": index,
            "pid": index + 100,
            "windows": [
                {
                    "concurrency": c,
                    "window": name,
                    "resource_telemetry_complete": False,
                    "metrics": {"mean_ms": 10.0, "p95_ms": 12.0, "rps": 100.0},
                }
                for c in probe.CONCURRENCIES
                for name, _ in probe.WINDOWS
            ],
        }
        for index in range(probe.PROCESSES)
    ]


def test_symmetric_stability_and_between_process_dispersion() -> None:
    records = process_rows()
    result = probe.analyze(records)
    assert result["statistical_stability_passed"]
    assert not result["qualification"]
    assert not result["resource_telemetry_complete"]
    assert sum(len(value) for value in result["directional_gates"].values()) == 12
    for row in records:
        for window in row["windows"]:
            if window["window"] == "L2":
                window["metrics"]["mean_ms"] *= 0.9
    result = probe.analyze(records)
    gates = result["directional_gates"]["C1_mean_ms"]
    assert gates["L2_over_L1_degradation"]["passed"]
    assert not gates["L1_over_L2_degradation"]["passed"]
    assert not result["statistical_stability_passed"]
    assert result["between_process_dispersion"]["C1_mean_ms"]["L2"]["values_by_process"] == [9.0] * 6


@pytest.mark.parametrize(
    "failure", ["missing_process", "same_pid", "missing_window", "order", "cleanup", "negative", "uncertain"]
)
def test_all_fixed_process_units_and_positive_metrics_required(failure: str) -> None:
    rows = process_rows()
    if failure == "missing_process":
        rows.pop()
    elif failure == "same_pid":
        rows[-1]["pid"] = rows[0]["pid"]
    elif failure == "missing_window":
        rows[-1]["windows"].pop()
    elif failure == "order":
        rows[-1]["windows"].reverse()
    elif failure == "cleanup":
        rows[-1]["shutdown_complete"] = False
    elif failure == "negative":
        for row in rows:
            row["windows"][1]["metrics"]["mean_ms"] = -10
            row["windows"][2]["metrics"]["mean_ms"] = -10
    else:
        for row, ratio in zip(rows, [0.94, 0.94, 0.94, 1.06, 1.06, 1.06], strict=True):
            row["windows"][2]["metrics"]["mean_ms"] *= ratio
        assert not probe.analyze(rows)["statistical_stability_passed"]
        return
    with pytest.raises(AssertionError):
        probe.analyze(rows)


def test_optional_cgroup_unavailable_is_explicit_not_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(process_metrics, "unified_cgroup", lambda: {"path": None, "error": "not exposed"})
    first, last = probe.snapshot(), probe.snapshot()
    delta = probe.telemetry_delta(first, last)
    assert set(first["cgroup_files"]) == set(process_metrics.OPTIONAL_FILES)
    assert all(value is None for value in first["cgroup_files"].values())
    assert delta["optional_cgroup"]["cpu.stat"] is None
    assert not delta["resource_telemetry_complete"]
    assert delta["optional_errors_before"]["cgroup"] == "not exposed"


@pytest.mark.parametrize("failure", ["gc_missing", "gc_count", "rusage", "clock", "disabled"])
def test_required_telemetry_validation(failure: str) -> None:
    core = observed()["core"]
    if failure == "gc_missing":
        core["gc_stats"] = []
    elif failure == "gc_count":
        core["gc_count"] = [-1, 0, 0]
    elif failure == "rusage":
        core["rusage"]["ru_utime"] = float("nan")
    elif failure == "clock":
        core["wall_ns"] = 0
    else:
        core["gc_enabled"] = False
    with pytest.raises(AssertionError):
        process_metrics.validate_core(core)


def test_own_cgroup_resolution_and_no_namespace_fallback(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    mount = tmp_path / "mounted"
    owned = mount / "child"
    owned.mkdir(parents=True)
    (proc / "cgroup").write_text("0::/namespace/child\n")
    line = f"1 0 0:1 /namespace {mount} rw - cgroup2 cgroup rw\n"
    (proc / "mountinfo").write_text(line)
    assert process_metrics.unified_cgroup(proc)["path"] == str(owned)
    (proc / "mountinfo").write_text(line + line)
    assert process_metrics.unified_cgroup(proc)["path"] is None
    (proc / "cgroup").write_text("0::/other/child\n")
    (proc / "mountinfo").write_text(line)
    assert process_metrics.unified_cgroup(proc)["path"] is None


def test_exposed_counters_and_gauges_have_distinct_semantics() -> None:
    first, last = observed(1), observed(2)
    for record, ticks, avg in ((first, 100, 90), (last, 120, 1)):
        record["cgroup"] = {"path": "/owned", "device": 1, "inode": 2, "error": None}
        record["cgroup_files"]["cpu.stat"] = process_metrics.flat_counters(f"usage_usec {ticks}\nnr_throttled 1\n")
        record["cgroup_files"]["cpu.pressure"] = process_metrics.pressure(
            f"some avg10={avg} avg60=0 avg300=0 total={ticks}\n"
        )
    last["core"]["gc_count"] = [0, 0, 0]  # Automatic GC resets this gauge normally.
    delta = probe.telemetry_delta(first, last)
    assert delta["cpu_pressure_counters_available"]
    assert not delta["resource_telemetry_complete"]  # Other optional files remain unavailable.
    assert delta["optional_cgroup"]["cpu.pressure"]["some"] == 20
    assert delta["optional_cgroup"]["cpu.stat"]["usage_usec"] == 20
    last["cgroup_files"]["cpu.pressure"]["some"]["total"] = 99
    with pytest.raises(AssertionError, match="reset"):
        probe.telemetry_delta(first, last)
    for text in ("usage_usec -1", "usage_usec 0\nusage_usec 1", ""):
        with pytest.raises((AssertionError, ValueError)):
            process_metrics.flat_counters(text)
    with pytest.raises(AssertionError, match="PSI"):
        process_metrics.pressure("some avg10=nan avg60=0 avg300=0 total=1")


@pytest.mark.parametrize(
    "changed", ["path", "inode", "visibility", "affinity", "quota", "gc_thresholds", "rusage", "gc_stats"]
)
def test_telemetry_continuity_refuses_changes(changed: str) -> None:
    first, last = observed(2), observed(3)
    first["cgroup"] = {"path": "/owned", "device": 1, "inode": 2, "error": None}
    last["cgroup"] = copy.deepcopy(first["cgroup"])
    if changed in {"path", "inode"}:
        last["cgroup"][changed] = "changed"
    elif changed == "visibility":
        last["cgroup"] = {"path": None, "error": "lost"}
    elif changed == "affinity":
        last["affinity"] = [0]
    elif changed == "quota":
        last["cgroup_files"]["cpu.max"] = "800000 100000"
    elif changed == "gc_thresholds":
        last["core"]["gc_thresholds"][0] = 800
    elif changed == "rusage":
        last["core"]["rusage"]["ru_nvcsw"] = 1
    else:
        last["core"]["gc_stats"][0]["collections"] = 1
    with pytest.raises(AssertionError):
        probe.telemetry_delta(first, last)


def response(label: str, index: int) -> dict[str, Any]:
    scores = [4.0, 3.0, 2.0, 1.0]
    return {
        "label": label,
        "selection": None,
        "start_ns": index * 10000000 + 1,
        "end_ns": (index + 1) * 10000000,
        "latency_ms": 10.0,
        "scores": scores,
        "responses": [{"status": 200, "timing_headers": {"x-inference-time": "4.0"}}],
        "result": {
            "scores": [{"item_id": f"{label}:{i}", "rank": i, "score": score} for i, score in enumerate(scores)]
        },
    }


def stub_performance(_url: str, concurrency: int, evidence: probe.DiagnosticEvidence, *, smoke: bool) -> dict[str, Any]:
    warmups = [response(f"warm-{i}", i) for i in range(2 if smoke else worker.WARMUP)]
    samples = [response(f"sample-{i}", i) for i in range(4 if smoke else worker.SEQUENTIAL)]
    elapsed = 0.2 if smoke else float(worker.STEADY_SECONDS)
    evidence.emit(
        "performance_attempt",
        concurrency=concurrency,
        warmups=warmups,
        completed_samples=samples,
        wire_responses=[row["responses"][0] for row in [*warmups, *samples]],
    )
    return {
        "concurrency": concurrency,
        "count": len(samples),
        "elapsed_s": elapsed,
        "mean_ms": 10.0,
        "p95_ms": 10.0,
        "rps": len(samples) / elapsed,
        "samples": samples,
    }


def install_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    counter = iter(range(1, 100))
    monkeypatch.setattr(probe, "snapshot", lambda: observed(next(counter)))
    monkeypatch.setattr(worker, "telemetry", lambda: {"device": "test"})
    monkeypatch.setattr(worker, "performance", stub_performance)


def current_workload() -> tuple[Any, ...]:
    return tuple(
        getattr(worker, name, None)
        for name in ("WARMUP", "SEQUENTIAL", "STEADY_SECONDS", "PERFORMANCE_FUTURE_TIMEOUT_S")
    )


def test_raw_retention_reopen_and_workload_restoration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_stub(monkeypatch)
    monkeypatch.setattr(worker, "PERFORMANCE_FUTURE_TIMEOUT_S", 120, raising=False)
    original = current_workload()
    continuity = observed(0)
    continuity["core"]["wall_ns"] = continuity["core"]["monotonic_ns"] = 1
    oracles = {"shape": {"copies": 1, "scores": [[4.0, 3.0, 2.0, 1.0][index] for index in worker.ORDER]}}
    compact = probe.measure_window(
        "unused", 1, "L1", 200, tmp_path / "C1/L1", [4.0, 3.0, 2.0, 1.0], oracles, smoke=False, continuity=continuity
    )
    assert original == current_workload()
    raw = probe.reopen_window(tmp_path, compact)
    assert len(raw["performance"]["samples"]) == 200
    assert raw["measurement_method"]["concurrent_measured_seconds"] == 12
    assert raw["measurement_method"]["future_wait_seconds"] == 90
    assert len(raw["chronological_quartiles"]) == 4
    assert continuity == raw["telemetry_after"]
    assert all(row["diagnostic_matching_oracle"]["workload_sha256"] == "shape" for row in raw["performance"]["samples"])
    assert "samples" not in json.dumps(compact)
    tampered = copy.deepcopy(compact)
    tampered["metrics"]["mean_ms"] = 9.0
    with pytest.raises(AssertionError, match="projection"):
        probe.reopen_window(tmp_path, tampered)
    tampered = {**compact, "record": "../outside.json"}
    with pytest.raises(AssertionError, match="path"):
        probe.reopen_window(tmp_path, tampered)
    (tmp_path / "C1/L1/events.jsonl").write_text("{}\n")
    with pytest.raises(AssertionError, match="attempt hash"):
        probe.reopen_window(tmp_path, compact)


def test_failed_window_keeps_partial_raw_and_restores_workload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_stub(monkeypatch)
    monkeypatch.setattr(worker, "PERFORMANCE_FUTURE_TIMEOUT_S", 120, raising=False)
    original = current_workload()

    def failed(_url: str, _concurrency: int, evidence: probe.DiagnosticEvidence, *, smoke: bool) -> None:
        assert current_workload() == (200, 200, 12, 90)
        assert not smoke
        evidence.emit("performance_attempt", completed_samples=[{"error": "retained"}])
        raise RuntimeError("request failed")

    monkeypatch.setattr(worker, "performance", failed)
    with pytest.raises(RuntimeError, match="request failed"):
        probe.measure_window("unused", 1, "L2", 200, tmp_path / "C1/L2", [4.0, 3.0, 2.0, 1.0], {}, smoke=False)
    assert original == current_workload()
    terminal = json.loads((tmp_path / "C1/L2/window.json").read_text())
    assert not terminal["qualification"]
    assert not terminal["completed"]
    assert terminal["error"]["message"] == "request failed"
    assert json.loads((tmp_path / "C1/L2/events.jsonl").read_text())["completed_samples"] == [{"error": "retained"}]


def test_diagnostic_settings_restore_absent_attribute(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(worker, "PERFORMANCE_FUTURE_TIMEOUT_S", raising=False)
    with probe.diagnostic_settings(30):
        assert current_workload() == (30, 200, 12, 90)
    assert not hasattr(worker, "PERFORMANCE_FUTURE_TIMEOUT_S")


def test_diagnostic_records_current_observer_and_exact_historical_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker, "RESPONSE_CAPTURE_MODE", "success-status-header-snapshots", raising=False)
    method = probe.measurement_method()
    assert method["response_capture"] == "success-status-header-snapshots"
    assert method["worker_source_sha256"] == digest(Path(worker.__file__))
    assert method["exact_historical_replay_commit"] == "482b599224920a97bb2fce4a740317efb89351ab"
    assert not method["historical_byte_identity_claimed"]


def test_full_cleanup_budget_is_strict() -> None:
    assert probe.admission(1, 272, 180)["reserve_s"] == 270
    with pytest.raises(AssertionError, match="insufficient"):
        probe.admission(1, 271, 180)


def test_negative_raw_latency_cannot_hide_in_positive_aggregates() -> None:
    phase = {
        "samples": [{"latency_ms": value} for value in [-1.0, 2.0, 3.0, 4.0]],
        "elapsed_s": 1.0,
        "count": 4,
        "mean_ms": 2.0,
        "p95_ms": 4.0,
        "rps": 4.0,
    }
    with pytest.raises(AssertionError, match="raw latency"):
        probe.metrics(phase)


def test_memory_and_observer_flow_preserved_with_explicit_production_class() -> None:
    baseline = subprocess.run(
        ["git", "show", "a2932d81:tools/benchmarks/cross_encoder_lora/worker.py"],  # noqa: S607
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if baseline.returncode:
        pytest.skip("invariant control requires its documented parent commit in git history")

    def code(text: str) -> dict[str, str]:
        class ClassBinding(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id == "production":
                    node.id = "CrossEncoderAdapter"
                return node

        tree = ast.parse(text)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "unobserved":
                binding = ast.parse("production = adapter.production_adapter_class").body[0]
                node.body = [entry for entry in node.body if ast.dump(entry) != ast.dump(binding)]
                ClassBinding().visit(node)
        return {
            node.name: ast.dump(node)
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in {"unobserved", "memory"}
        }

    assert code(Path(worker.__file__).read_text()) == code(baseline.stdout)
    # The explicit opt-in subclass now supplies selection; all timing/restore
    # control flow and memory semantics stay the same. Real R forwarding is
    # checked separately, including refusal of a poisoned resident manager.
    # Historical diagnostic timing still uses the in-process wrapper and does
    # not force GC. The new qualification load loop has its own extraction and
    # clock-boundary tests; exact historical replay remains pinned to 482b.
    tree = ast.parse(Path(probe.__file__).read_text())
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "gc"
        and node.func.attr in {"collect", "disable", "set_threshold"}
        for node in ast.walk(tree)
    )


def test_cross_process_source_and_environment_manifest(tmp_path: Path) -> None:
    (tmp_path / "module.py").write_text("# immutable baseline fixture\n")
    distribution = [["torch", "exact"]]
    environment = {
        "qualification": False,
        "versions": {},
        "torchao": None,
        "python": "fixed",
        "threads": 4,
        "device": "cpu",
        "torch_cuda": None,
        "hardware_preflight_only": False,
        "installed_distributions": distribution,
        "installed_distributions_sha256": fingerprint(distribution),
        "source_imports": {
            name: {"relative_path": "module.py", "sha256": digest(tmp_path / "module.py")}
            for name in ("sdk", "server", "adapter")
        },
    }
    row: dict[str, Any] = {
        "source_commit": BASE_REVISION,
        "smoke": True,
        "environment": environment,
        "fixtures": {},
        "actual_precision": "float32",
        "base_scores": [1.0],
        "shape_oracles": {"a": {"scores": [1.0]}},
        "required_telemetry_preflight": observed(),
    }
    other = copy.deepcopy(row)
    probe.verify_process_sources([row, other], tmp_path)
    other["environment"]["installed_distributions"] = [["torch", "different"]]
    other["environment"]["installed_distributions_sha256"] = fingerprint(
        other["environment"]["installed_distributions"]
    )
    with pytest.raises(AssertionError, match="environments differ"):
        probe.verify_process_sources([row, other], tmp_path)
    other = copy.deepcopy(row)
    other["required_telemetry_preflight"]["affinity"] = [0]
    with pytest.raises(AssertionError, match="affinity differs"):
        probe.verify_process_sources([row, other], tmp_path)
    (tmp_path / "module.py").write_text("# changed\n")
    with pytest.raises(AssertionError, match="imported source differs"):
        probe.verify_process_sources([row], tmp_path)


@pytest.fixture(scope="module")
def real_inputs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    directory = tmp_path_factory.mktemp("stability-real")
    available = subprocess.run(["git", "cat-file", "-e", BASE_REVISION], cwd=ROOT, capture_output=True, check=False)  # noqa: S607
    if available.returncode:
        pytest.skip("immutable B smoke requires the documented baseline in git history")
    snapshot = tiny_snapshot(directory / "model")
    fixtures, inputs = directory / "fixtures", directory / "inputs"
    prepare(fixtures, snapshot, smoke=True)
    source_preparer.prepare(ROOT, inputs, "HEAD")
    return directory, fixtures, inputs


def test_real_immutable_baseline_nine_window_sdk_smoke(real_inputs: tuple[Path, Path, Path]) -> None:
    directory, fixtures, inputs = real_inputs
    output = directory / "diagnostic"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.benchmarks.cross_encoder_lora.stability",
            "--inputs",
            str(inputs),
            "--fixtures",
            str(fixtures),
            "--output",
            str(output),
            "--smoke",
            "--deadline-s",
            "300",
            "--trial-deadline-s",
            "90",
        ],
        cwd=ROOT,
        env=source_environment(ROOT),
        capture_output=True,
        text=True,
        timeout=150,
        check=False,
    )
    summary = json.loads((output / "summary.json").read_text())
    assert result.returncode == 0, result.stdout + result.stderr + (output / "process-0.log").read_text()
    assert summary["completed"]
    assert not summary["qualification"]
    assert not summary["statistical_stability_passed"]
    process = json.loads((output / "process-0/process.json").read_text())
    assert process["shutdown_complete"]
    assert not process["environment"]["qualification"]
    assert len(process["windows"]) == 9
    assert process["source_commit"] == BASE_REVISION
    for row in process["windows"]:
        window = probe.reopen_window(output / "process-0", row)
        assert window["warmups"] == 2
        assert not window["qualification"]
    assert all(batch["forwards"] for replay in process["untimed_replays"].values() for batch in replay["batches"])
    probe.verify_process_sources([process], output / "sources/B")


def test_required_preflight_and_live_failure_cleanup(real_inputs: tuple[Path, Path, Path]) -> None:
    directory, fixtures, inputs = real_inputs
    sources = directory / "failure-sources"
    extract(inputs, sources)
    source = sources / "B"
    for failure in ("preflight", "window"):
        output = directory / f"failed-{failure}"
        script = """
import argparse, asyncio, json
from pathlib import Path
from tools.benchmarks.cross_encoder_lora import stability as s
def fail(*args, **kwargs):
    raise RuntimeError('injected diagnostic failure')
args = argparse.Namespace(source=Path(SOURCE), fixtures=Path(FIXTURES), output=Path(OUTPUT), process=0,
                          smoke=True, trial_deadline_s=90)
if FAILURE == 'preflight':
    s.snapshot = fail
    s.serve_windows = lambda *args: (_ for _ in ()).throw(AssertionError('model work before telemetry'))
else:
    s.measure_window = fail
asyncio.run(s.child(args))
"""
        bindings = f"SOURCE={str(source)!r}\nFIXTURES={str(fixtures)!r}\nOUTPUT={str(output)!r}\nFAILURE={failure!r}\n"
        result = subprocess.run(
            [sys.executable, "-c", bindings + script],
            cwd=source,
            env=source_environment(source),
            capture_output=True,
            text=True,
            timeout=130,
            check=False,
        )
        assert result.returncode != 0
        record = json.loads((output / "process.json").read_text())
        assert not record["completed"]
        assert not record["qualification"]
        assert record["error"]["message"] == "injected diagnostic failure", result.stdout + result.stderr
        assert record["shutdown_complete"] == (failure == "window")
        if failure == "preflight":
            assert "environment" not in record
        else:
            assert record["untimed_replays"]["16"]["batches"]
