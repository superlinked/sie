from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tools.ci import required_ci

ROOT = Path(__file__).resolve().parents[3]


def successful_needs():
    return {name: {"result": "success"} for name in required_ci.MANDATORY_JOBS}


def test_every_mandatory_lane_must_succeed():
    assert required_ci.failures(successful_needs()) == []


@pytest.mark.parametrize("name", required_ci.MANDATORY_JOBS)
@pytest.mark.parametrize("result", ["failure", "cancelled", "skipped", "neutral", "timed_out", "", None])
def test_any_non_success_result_is_rejected(name, result):
    needs = successful_needs()
    needs[name]["result"] = result
    assert required_ci.failures(needs) == [f"{name}: {result}"]


@pytest.mark.parametrize("name", required_ci.MANDATORY_JOBS)
def test_missing_lane_is_rejected(name):
    needs = successful_needs()
    del needs[name]
    assert required_ci.failures(needs) == [f"{name}: missing"]


@pytest.mark.parametrize(("result", "code"), [("success", 0), ("skipped", 1), ("cancelled", 1), ("failure", 1)])
def test_gate_process_exit_status(result, code):
    needs = successful_needs()
    needs["python"]["result"] = result
    completed = subprocess.run(
        [sys.executable, str(ROOT / "tools/ci/required_ci.py")],
        env={**os.environ, "NEEDS": json.dumps(needs)},
        check=False,
        capture_output=True,
    )
    assert completed.returncode == code


def test_ci_mandatory_graph_and_permissions():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    jobs = workflow["jobs"]
    assert set(jobs) == {*required_ci.MANDATORY_JOBS, "required"}
    assert set(jobs["required"]["needs"]) == set(required_ci.MANDATORY_JOBS)
    assert jobs["required"]["if"] == "${{ always() }}"
    assert workflow["permissions"] == {"contents": "read"}
    for name in required_ci.MANDATORY_JOBS:
        assert "if" not in jobs[name]
        assert "environment" not in jobs[name]
        assert "secrets" not in jobs[name]
        if "runs-on" in jobs[name]:
            assert re.fullmatch(r"blacksmith-[248]vcpu-ubuntu-2404", jobs[name]["runs-on"])
        for step in jobs[name].get("steps", []):
            if "uses" in step:
                assert re.fullmatch(r"[\w/-]+@[a-f0-9]{40}", step["uses"])
    serialized = json.dumps(workflow)
    for forbidden in ("pull_request_target", "id-token", "secrets.", "classify_paths", "BENCHMARK"):
        assert forbidden not in serialized


def test_bootstrap_is_uncached_and_checks_all_locks():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    bootstrap = workflow["jobs"]["bootstrap"]
    setup = next(step for step in bootstrap["steps"] if step.get("uses", "").startswith("jdx/mise-action@"))
    assert setup["with"] == {"cache": False, "install": False}
    script = (ROOT / "tools/ci/fresh_bootstrap.bash").read_text()
    assert "./tools/init.sh" in script
    assert "test ! -e .venv" in script
    assert "test ! -e node_modules" in script
    for lock in ("uv.lock", "pnpm-lock.yaml", "Cargo.lock"):
        assert lock in script
    assert "sha256sum --check" in script


@pytest.mark.parametrize(("mutate_lock", "old_venv", "code"), [(False, False, 0), (True, False, 1), (False, True, 1)])
def test_bootstrap_rejects_reused_environment_and_changed_lock(tmp_path, mutate_lock, old_venv, code):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    for lock in ("uv.lock", "pnpm-lock.yaml", "Cargo.lock"):
        (tmp_path / lock).write_text("committed-lock\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "uv.lock", "pnpm-lock.yaml", "Cargo.lock"], check=True)
    (tmp_path / "tools").mkdir()
    init = tmp_path / "tools/init.sh"
    init.write_text("#!/bin/sh\n" + ("printf changed > uv.lock\n" if mutate_lock else ":\n"))
    init.chmod(0o755)
    if old_venv:
        (tmp_path / ".venv").mkdir()
    result = subprocess.run(
        ["bash", str(ROOT / "tools/ci/fresh_bootstrap.bash")], cwd=tmp_path, capture_output=True, check=False
    )
    assert result.returncode == code


def test_typescript_build_precedes_typecheck():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    commands = [step.get("run") for step in workflow["jobs"]["typescript"]["steps"]]
    assert commands.index("mise run ts -- build") < commands.index("mise run ts -- typecheck")
