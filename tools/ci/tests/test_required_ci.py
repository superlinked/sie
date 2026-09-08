from __future__ import annotations

import fnmatch
import json
import os
import re
import shlex
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
    assert setup["with"] == {"version": "2026.5.5", "cache": False, "install": False}
    script = (ROOT / "tools/ci/fresh_bootstrap.bash").read_text()
    assert "./tools/init.sh" in script
    assert "test ! -e .venv" in script
    assert "test ! -e node_modules" in script
    for lock in ("uv.lock", "pnpm-lock.yaml", "Cargo.lock"):
        assert lock in script
    assert "sha256sum --check" in script


def test_mise_workflow_setups_pin_concrete_versions():
    for path in sorted((ROOT / ".github/workflows").glob("*.y*ml")):
        workflow = yaml.safe_load(path.read_text())
        for name, job in workflow["jobs"].items():
            for step in job.get("steps", []):
                if step.get("uses", "").startswith("jdx/mise-action@"):
                    version = step.get("with", {}).get("version")
                    assert re.fullmatch(r"[0-9]{4}\.[0-9]+\.[0-9]+", str(version)), (
                        f"{path.name}: {name} must pin a concrete mise version"
                    )


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


def test_contracts_regenerate_both_openapi_documents_before_exact_diff():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    commands = [step.get("run", "") for step in workflow["jobs"]["contracts"]["steps"]]
    generation = commands.index("mise run openapi")
    diff = next(
        index
        for index, command in enumerate(commands)
        if "git diff --exit-code" in command and "packages/sie_server/openapi.json" in command
    )
    assert generation < diff
    assert shlex.split(commands[diff]) == [
        "git",
        "diff",
        "--exit-code",
        "--",
        "packages/sie_server/openapi.json",
        "packages/sie_gateway/openapi.json",
    ]


def test_helm_renders_each_cloud_overlay():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    commands = [step.get("run", "") for step in workflow["jobs"]["helm"]["steps"]]
    assert "mise run helm -- template --set payloadStore.enabled=false >/dev/null" in commands
    overlay_command = next(command for command in commands if "for values in" in command)
    for overlay in ("values-aws.yaml", "values-gke.yaml", "values-aks.yaml", "values-ack.yaml"):
        assert overlay_command.count(overlay) == 1
    assert "--set payloadStore.enabled=false" in overlay_command


def test_cpu_lane_uses_cpu_stack_task():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    commands = [step.get("run") for step in workflow["jobs"]["cpu-stack"]["steps"] if "run" in step]
    assert commands == ["mise run cpu-stack"]


def test_rust_audits_both_committed_dependency_graphs():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    commands = [shlex.split(step.get("run", "")) for step in workflow["jobs"]["rust"]["steps"]]
    assert ["mise", "run", "gateway-deny"] in commands
    assert [
        "mise",
        "exec",
        "--",
        "cargo-deny",
        "--locked",
        "--manifest-path",
        "packages/sie_server_rust/Cargo.toml",
        "--all-features",
        "--config",
        "deny.toml",
        "check",
    ] in commands


CUDA13_SHARED_PATHS = {
    ".github/workflows/cuda13-bundle-image.yml",
    "tools/ci/cuda13_image_smoke.py",
    "Cargo.toml",
    "Cargo.lock",
    "packages/sie_audio_prep/**",
    "packages/sie_gateway/**",
    "packages/sie_server_sidecar/**",
    "packages/sie_telemetry/**",
    "packages/sie_sdk/pyproject.toml",
    "packages/sie_sdk/src/**",
    "packages/sie_server/Dockerfile.cuda13",
    "packages/sie_server/src/**",
    "packages/sie_server/bundles/**",
    "packages/sie_server/models/**",
    "packages/sie_server/pyproject.toml",
}

CUDA_CALLERS = {
    "cuda13-sglang-cu130.yml": {
        "bundle": "sglang-cu130",
        "image-tag": "sie-cuda13-sglang-cu130:pr",
        "paths": CUDA13_SHARED_PATHS
        | {
            ".github/workflows/cuda13-sglang-cu130.yml",
            "packages/sie_server/bundles/sglang-cu130.yaml",
            "packages/sie_server/models/Qwen__Qwen3.6-27B.yaml",
            "packages/sie_server/models/Qwen__Qwen3.8-27B-FP8.yaml",
            "packages/sie_server/models/google__gemma-4-*.yaml",
            "packages/sie_server/src/sie_server/adapters/sglang/**",
        },
    },
    "cuda13-tensorrt-llm.yml": {
        "bundle": "tensorrt-llm",
        "image-tag": "sie-cuda13-tensorrt-llm:pr",
        "paths": CUDA13_SHARED_PATHS
        | {
            ".github/workflows/cuda13-tensorrt-llm.yml",
            "packages/sie_server/bundles/tensorrt-llm.yaml",
            "packages/sie_server/src/sie_server/adapters/tensorrt_llm/**",
        },
    },
}


@pytest.mark.parametrize(("filename", "expected"), CUDA_CALLERS.items())
def test_cuda13_callers_have_exact_inputs_and_narrow_paths(filename, expected):
    workflow = yaml.safe_load((ROOT / ".github/workflows" / filename).read_text())
    triggers = workflow.get("on", workflow.get(True))
    assert set(triggers) == {"pull_request"}
    assert set(triggers["pull_request"]["paths"]) == expected["paths"]
    assert workflow["permissions"] == {"contents": "read"}
    assert set(workflow["jobs"]) == {"compatibility"}
    job = workflow["jobs"]["compatibility"]
    assert job["uses"] == "./.github/workflows/cuda13-bundle-image.yml"
    assert job["with"] == {"bundle": expected["bundle"], "image-tag": expected["image-tag"]}


def test_sglang_cuda13_model_configs_are_covered_by_caller_paths():
    workflow = yaml.safe_load((ROOT / ".github/workflows/cuda13-sglang-cu130.yml").read_text())
    triggers = workflow.get("on", workflow.get(True))
    model_patterns = [
        path for path in triggers["pull_request"]["paths"] if path.startswith("packages/sie_server/models/")
    ]
    adapter_modules = ("sie_server.adapters.sglang.cuda13", "sie_server.adapters.sglang.gemma")
    referenced_models = {
        str(path.relative_to(ROOT))
        for path in (ROOT / "packages/sie_server/models").glob("*.yaml")
        if any(module in path.read_text() for module in adapter_modules)
    }
    assert referenced_models
    assert all(any(fnmatch.fnmatch(model, pattern) for pattern in model_patterns) for model in referenced_models)


@pytest.mark.parametrize(
    ("filename", "group"),
    [
        (
            "cuda13-sglang-cu130.yml",
            "cuda13-sglang-cu130-${{ github.event.pull_request.number || github.ref }}",
        ),
        (
            "cuda13-tensorrt-llm.yml",
            "cuda13-tensorrt-llm-${{ github.event.pull_request.number || github.ref }}",
        ),
    ],
)
def test_cuda13_callers_cancel_superseded_pr_runs(filename, group):
    workflow = yaml.safe_load((ROOT / ".github/workflows" / filename).read_text())
    assert workflow["concurrency"] == {"group": group, "cancel-in-progress": True}


def test_cuda13_reusable_workflow_is_read_only_pinned_and_build_only():
    workflow = yaml.safe_load((ROOT / ".github/workflows/cuda13-bundle-image.yml").read_text())
    triggers = workflow.get("on", workflow.get(True))
    inputs = triggers["workflow_call"]["inputs"]
    assert inputs == {
        "bundle": {"required": True, "type": "string"},
        "image-tag": {"required": True, "type": "string"},
    }
    assert workflow["permissions"] == {"contents": "read"}
    job = workflow["jobs"]["compatibility"]
    assert job["runs-on"] == "blacksmith-8vcpu-ubuntu-2404"
    assert job["timeout-minutes"] == 90
    assert [step.get("uses") for step in job["steps"] if "uses" in step] == [
        "actions/checkout@93cb6efe18208431cddfb8368fd83d5badbf9bfd",
        "useblacksmith/setup-docker-builder@9309da73a81f66976a6d750572e221508b1e2682",
        "useblacksmith/build-push-action@9b0579bbec7a6cad2f171596c57e7ac1e7658850",
    ]
    build = next(step for step in job["steps"] if step.get("uses", "").startswith("useblacksmith/build-push-action@"))
    assert build["with"] == {
        "context": ".",
        "file": "packages/sie_server/Dockerfile.cuda13",
        "platforms": "linux/amd64",
        "build-args": "BUNDLE=${{ inputs.bundle }}\nSIE_SRC_REV=${{ github.sha }}\n",
        "tags": "${{ inputs.image-tag }}",
        "load": True,
        "push": False,
    }
    assert job["steps"][-1]["run"] == (
        'python3 tools/ci/cuda13_image_smoke.py "${{ inputs.bundle }}" "${{ inputs.image-tag }}"'
    )


def test_cuda13_workflows_have_no_privileged_or_unrelated_behavior():
    for filename in ("cuda13-bundle-image.yml", *CUDA_CALLERS):
        text = (ROOT / ".github/workflows" / filename).read_text()
        lowered = text.lower()
        for forbidden in (
            "pull_request_target",
            "secrets.",
            "docker login",
            "environment:",
            "docker push",
            "benchmark",
            "quality",
            "load-test",
        ):
            assert forbidden not in lowered
        workflow = yaml.safe_load(text)
        for job in workflow["jobs"].values():
            uses = [job.get("uses", ""), *(step.get("uses", "") for step in job.get("steps", []))]
            for action in filter(None, uses):
                assert action.startswith("./") or re.fullmatch(r"[\w/-]+@[a-f0-9]{40}", action)
