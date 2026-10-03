from __future__ import annotations

import fnmatch
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
FAMILIES = ("python", "npm", "docker", "audio", "helm", "native")


def workflow() -> dict:
    return yaml.safe_load((ROOT / ".github/workflows/release-candidate.yml").read_text())


def step_python(job: str) -> str:
    run = workflow()["jobs"][job]["steps"][-1]["run"]
    match = re.fullmatch(r"python3 - <<'PY'\n(.*)\nPY\n?", run, re.DOTALL)
    assert match is not None
    return match[1]


def run_step(job: str, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - execute the checked-in candidate workflow script
        [sys.executable, "-c", step_python(job)],
        cwd=ROOT,
        env={**os.environ, **environment},
        capture_output=True,
        text=True,
        check=False,
    )


def test_candidate_uses_every_pure_builder_at_one_source_without_publication():
    candidate = workflow()
    triggers = candidate.get("on", candidate.get(True))
    assert set(triggers) == {"pull_request", "workflow_dispatch"}
    for path in (
        ".github/workflows/release-candidate.yml",
        ".github/workflows/release.yml",
        ".github/workflows/publish-docker.yml",
        ".github/release-matrix.json",
        ".release-please-manifest.json",
        "tools/ci/rust_cuda_image_smoke.py",
        "tools/mise_tasks/docker_task.py",
        "tools/mise_tasks/helm.py",
        "packages/sie_server/Dockerfile.cuda13",
        "packages/sie_audio_prep/build_wheel.py",
        "packages/sie_server_rust/Cargo.lock",
        "deploy/helm/sie-cluster/Chart.yaml",
    ):
        assert any(fnmatch.fnmatchcase(path, pattern) for pattern in triggers["pull_request"]["paths"]), path
    assert candidate["permissions"] == {"contents": "read"}
    jobs = candidate["jobs"]
    assert set(jobs) == {"prepare", *FAMILIES, "complete"}
    assert jobs["prepare"]["steps"][0]["with"] == {"ref": "${{ github.sha }}", "persist-credentials": False}
    for family in FAMILIES:
        job = jobs[family]
        assert job["uses"] == f"./.github/workflows/release-{family}.yml"
        assert job["needs"] == (["prepare", "docker"] if family == "native" else "prepare")
        expected: dict[str, str | bool] = {
            "version": "${{ needs.prepare.outputs.version }}",
            "tag_name": "${{ needs.prepare.outputs.tag_name }}",
        }
        expected["source_ref" if family in ("python", "npm") else "sha"] = "${{ needs.prepare.outputs.sha }}"
        if family in ("python", "npm"):
            expected["build_only"] = True
        assert job["with"] == expected
    for job in jobs.values():
        assert not {"environment", "secrets", "continue-on-error"} & job.keys()
        assert all(value == "read" for value in job.get("permissions", {}).values())
    assert jobs["docker"]["permissions"] == {"contents": "read", "actions": "read"}
    assert set(jobs["complete"]["needs"]) == {"prepare", *FAMILIES}
    assert jobs["complete"]["if"] == "${{ always() }}"
    for forbidden in (
        "pull_request_target",
        "id-token",
        "secrets.",
        "release-please-action",
        "docker push",
        "gh release",
    ):
        assert forbidden not in json.dumps(candidate)


def test_candidate_has_no_transitive_publisher_permissions():
    for family in FAMILIES:
        builder = yaml.safe_load((ROOT / f".github/workflows/release-{family}.yml").read_text())
        assert all(value == "read" for value in builder.get("permissions", {}).values())
        for job in builder["jobs"].values():
            assert not {"environment", "secrets"} & job.keys()
            assert all(value == "read" for value in job.get("permissions", {}).values())
        for forbidden in ("id-token", "secrets."):
            assert forbidden not in json.dumps(builder)


@pytest.mark.parametrize("wrong_source", [False, True])
def test_candidate_identity_reads_real_metadata_and_requires_exact_checkout(tmp_path, wrong_source):
    sha = subprocess.check_output(["/usr/bin/git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    output = tmp_path / "outputs"
    result = run_step("prepare", {"GITHUB_SHA": "0" * 40 if wrong_source else sha, "GITHUB_OUTPUT": str(output)})
    if wrong_source:
        assert result.returncode != 0
        assert not output.exists()
    else:
        assert result.returncode == 0, result.stderr
        version = json.loads((ROOT / ".release-please-manifest.json").read_text())["."]
        assert output.read_text() == f"sha={sha}\nversion={version}\ntag_name=v{version}\n"


@pytest.mark.parametrize("family", [None, "prepare", *FAMILIES])
def test_actual_candidate_completion_requires_all_families(family):
    needs = {name: {"result": "success"} for name in ("prepare", *FAMILIES)}
    if family is not None:
        needs[family]["result"] = "failure"
    completed = run_step("complete", {"RESULTS": json.dumps(needs)})
    assert (completed.returncode == 0) is (family is None), completed.stderr
