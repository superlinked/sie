from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import textwrap
from pathlib import Path

import pytest

from tools.ci import check_release_contract as contract


def run_audio_uploader(
    tmp_path: Path, *, asset_mode: str, remote_sha: str
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    tmp_path.mkdir()
    filename = "sie_audio_prep-0.7.4-cp312-abi3-manylinux_2_28_x86_64.whl"
    wheel = tmp_path / filename
    wheel.write_bytes(b"validated native wheel bytes")
    marker = tmp_path / "uploaded"
    calls = tmp_path / "calls"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    gh = fake_bin / "gh"
    gh.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
emit_asset() {
  printf '{"assets":[{"id":1,"name":"%s","size":%s,"digest":"sha256:%s","browser_download_url":"%s"}]}\\n' \\
    "$AUDIO_WHEEL_FILENAME" "$FAKE_REMOTE_SIZE" "$FAKE_REMOTE_SHA" "$FAKE_BROWSER_URL"
}
if [[ "$1" == api && "$2" == repos/*/releases/tags/* ]]; then
  if [[ "$FAKE_ASSET_MODE" == missing && ! -f "$FAKE_UPLOAD_MARKER" ]]; then
    printf '{"assets":[]}\\n'
  else
    emit_asset
  fi
elif [[ "$1" == release && "$2" == upload ]]; then
  printf '%s\\n' "$*" >> "$FAKE_CALLS"
  touch "$FAKE_UPLOAD_MARKER"
else
  printf 'unexpected fake gh arguments: %s\\n' "$*" >&2
  exit 2
fi
"""
    )
    gh.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "AUDIO_WHEEL_FILENAME": filename,
        "GITHUB_REPOSITORY": "superlinked/sie",
        "RELEASE_TAG": "v0.7.4",
        "GH_TOKEN": str(tmp_path),
        "FAKE_ASSET_MODE": asset_mode,
        "FAKE_REMOTE_SIZE": str(wheel.stat().st_size),
        "FAKE_REMOTE_SHA": remote_sha,
        "FAKE_BROWSER_URL": f"https://github.com/superlinked/sie/releases/download/v0.7.4/{filename}",
        "FAKE_UPLOAD_MARKER": str(marker),
        "FAKE_CALLS": str(calls),
    }
    result = subprocess.run(  # noqa: S603
        ["/bin/bash", str(contract.ROOT / "tools/ci/upload_audio_prep_release_asset.bash"), str(wheel)],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    return result, marker, calls


def test_exact_package_matrices() -> None:
    assert contract.python_matrices() == (
        contract.PYTHON_DISTRIBUTIONS,
        contract.PYTHON_DISTRIBUTIONS,
    )
    assert contract.npm_matrix() == contract.NPM_PACKAGES


def test_release_please_surface_is_exact() -> None:
    assert contract.release_config_errors() == []


def test_release_workflows_are_pinned_and_fail_closed() -> None:
    assert contract.workflow_pin_errors() == []
    assert contract.release_workflow_errors() == []
    assert contract.publisher_job_errors() == []


@pytest.mark.parametrize("replacement", ["  queue: single", "  queue: arbitrary", "", "  queue: max\n  queue: single"])
def test_release_queue_must_retain_pending_runs(replacement):
    top = (contract.ROOT / ".github/workflows/release.yml").read_text()
    ci = (contract.ROOT / ".github/workflows/ci.yml").read_text()
    assert contract.release_queue_errors(top, ci) == []
    assert contract.release_queue_errors(top.replace("  queue: max", replacement), ci)


def test_release_queue_cancellation_and_linter_exception_are_exact():
    top = (contract.ROOT / ".github/workflows/release.yml").read_text()
    ci = (contract.ROOT / ".github/workflows/ci.yml").read_text()
    assert contract.release_queue_errors(top.replace("cancel-in-progress: false", "cancel-in-progress: true"), ci)
    assert contract.release_queue_errors(top, ci.replace(contract.QUEUE_SCHEMA_DIAGNOSTIC, ".*"))
    diagnostic = 'unexpected key "queue" for "concurrency" section. expected one of "cancel-in-progress", "group"'
    assert re.fullmatch(contract.QUEUE_SCHEMA_DIAGNOSTIC, diagnostic)
    assert not re.fullmatch(contract.QUEUE_SCHEMA_DIAGNOSTIC, diagnostic.replace('"queue"', '"bogus"'))


def test_authoring_no_longer_assumes_release_sha_is_push_sha():
    jobs = contract.workflow_job_blocks(".github/workflows/release.yml")
    assert 'test "$RELEASE_SHA" = "$EXPECTED_SHA"' not in jobs["release-please"]
    assert "needs.release-please.outputs" not in (contract.ROOT / ".github/workflows/release.yml").read_text()
    assert "release_guard.py prepare" in jobs["prepare"]


def test_prereleases_are_ignored_by_prepare_and_completion():
    jobs = contract.workflow_job_blocks(".github/workflows/release.yml")
    for job in ("prepare", "complete"):
        condition = contract.job_scalar(jobs[job], "if")
        assert "github.event_name == 'release'" in condition
        assert "github.event.action == 'published'" in condition
        assert "github.event.release.draft == false" in condition
        assert "github.event.release.prerelease == false" in condition


@pytest.mark.parametrize("result", ["success", "failure", "cancelled", "skipped"])
def test_actual_release_completion_script_rejects_non_success(monkeypatch, result):
    block = contract.workflow_job_blocks(".github/workflows/release.yml")["complete"]
    script = re.search(r"python3 - <<'PY'\n(.*?)\n\s+PY", block, re.DOTALL)
    assert script is not None
    results = {
        family: {"result": "success"}
        for family in ("prepare", "python-publish", "npm-publish", "docker", "helm", "audio", "native")
    }
    results["native"]["result"] = result
    monkeypatch.setenv("RESULTS", json.dumps(results))
    code = compile(textwrap.dedent(script.group(1)), "release-complete", "exec")
    if result == "success":
        exec(code, {})  # noqa: S102
    else:
        with pytest.raises(SystemExit, match="Incomplete release"):
            exec(code, {})  # noqa: S102


def test_candle_and_docker_release_source_closure() -> None:
    assert contract.candle_source_errors() == []
    assert contract.docker_copy_errors() == []
    assert contract.docker_release_errors() == []


def test_helm_release_follows_verified_images() -> None:
    assert contract.helm_release_errors() == []


def test_public_release_app_hands_final_pr_head_to_ci() -> None:
    assert contract.release_app_errors() == []


def test_native_audio_asset_matches_downstream_browser_download_contract() -> None:
    version, filename, url = contract.audio_release_contract()
    assert filename == f"sie_audio_prep-{version}-cp312-abi3-manylinux_2_28_x86_64.whl"
    assert url == f"https://github.com/superlinked/sie/releases/download/v{version}/{filename}"
    assert contract.audio_release_errors() == []


def test_native_audio_builder_installs_exact_rust_and_never_clobbers() -> None:
    workflow = (contract.ROOT / ".github/workflows/release-audio.yml").read_text()
    uploader = (contract.ROOT / "tools/ci/upload_native_release_asset.bash").read_text()
    assert contract.AUDIO_MANYLINUX_IMAGE in workflow
    assert "version: 2026.7.11" in workflow
    assert "mise --no-config install python@3.12.12 uv@0.5.31 zig@0.13.0 rust@1.97.0" in workflow
    assert "rust@1.97.0 -- rustc --version" in workflow
    assert "rust@1.97.0 -- cargo --version" in workflow
    assert "--clobber" not in workflow
    assert "--clobber" not in uploader
    assert "sha256sum" in uploader


def test_native_audio_uploader_accepts_only_missing_or_identical_asset(tmp_path: Path) -> None:
    local_sha = hashlib.sha256(b"validated native wheel bytes").hexdigest()

    identical, identical_marker, _ = run_audio_uploader(
        tmp_path / "identical", asset_mode="present", remote_sha=local_sha
    )
    assert identical.returncode == 0, identical.stderr
    assert not identical_marker.exists()

    conflicting, conflicting_marker, _ = run_audio_uploader(
        tmp_path / "conflicting", asset_mode="present", remote_sha="0" * 64
    )
    assert conflicting.returncode != 0
    assert not conflicting_marker.exists()

    missing, missing_marker, calls = run_audio_uploader(
        tmp_path / "missing", asset_mode="missing", remote_sha=local_sha
    )
    assert missing.returncode == 0, missing.stderr
    assert missing_marker.is_file()
    assert calls.read_text().startswith("release upload --repo superlinked/sie v0.7.4 ")
    assert "--clobber" not in calls.read_text()
