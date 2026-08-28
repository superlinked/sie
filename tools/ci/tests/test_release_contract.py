from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from tools.ci import check_release_contract as contract


def run_audio_uploader(
    tmp_path: Path, *, asset_mode: str, remote_sha: str
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    tmp_path.mkdir()
    filename = "sie_audio_prep-0.7.2-cp312-abi3-manylinux_2_28_x86_64.whl"
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
        "RELEASE_TAG": "v0.7.2",
        "GH_TOKEN": str(tmp_path),
        "FAKE_ASSET_MODE": asset_mode,
        "FAKE_REMOTE_SIZE": str(wheel.stat().st_size),
        "FAKE_REMOTE_SHA": remote_sha,
        "FAKE_BROWSER_URL": f"https://github.com/superlinked/sie/releases/download/v0.7.2/{filename}",
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


def test_public_seed_tag_is_ancestral() -> None:
    assert contract.tag_errors() == []


def test_exported_tree_does_not_require_git_metadata(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(contract, "ROOT", tmp_path)
    assert contract.tag_errors() == []


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
    assert version == "0.7.2"
    assert filename == "sie_audio_prep-0.7.2-cp312-abi3-manylinux_2_28_x86_64.whl"
    assert url == (
        "https://github.com/superlinked/sie/releases/download/v0.7.2/"
        "sie_audio_prep-0.7.2-cp312-abi3-manylinux_2_28_x86_64.whl"
    )
    assert contract.audio_release_errors() == []


def test_native_audio_builder_installs_exact_rust_and_never_clobbers() -> None:
    workflow = (contract.ROOT / ".github/workflows/release-audio.yml").read_text()
    uploader = (contract.ROOT / "tools/ci/upload_audio_prep_release_asset.bash").read_text()
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
    assert calls.read_text().startswith("release upload --repo superlinked/sie v0.7.2 ")
    assert "--clobber" not in calls.read_text()


def test_stable_audio_repair_is_narrow_and_fail_closed() -> None:
    assert contract.repair_audio_errors() == []
    sha = "a" * 40
    base = {
        "latch": "true",
        "event_name": "workflow_dispatch",
        "ref": "refs/heads/main",
        "ref_protected": True,
        "repository": "superlinked/sie",
        "github_sha": sha,
        "main_sha": sha,
        "tag_sha": "b" * 40,
        "tag_name": "v0.7.2",
        "version": "0.7.2",
        "filename": "sie_audio_prep-0.7.2-cp312-abi3-manylinux_2_28_x86_64.whl",
        "tag_version": "0.7.2",
        "tag_filename": "sie_audio_prep-0.7.2-cp312-abi3-manylinux_2_28_x86_64.whl",
        "tag_is_ancestor": True,
        "release_is_stable": True,
    }
    assert contract.trusted_audio_repair_context(**base)
    for key, value in (
        ("latch", "false"),
        ("event_name", "pull_request"),
        ("ref", "refs/heads/feature"),
        ("ref_protected", False),
        ("repository", "someone/fork"),
        ("github_sha", "c" * 40),
        ("main_sha", "not-a-sha"),
        ("tag_sha", "not-a-sha"),
        ("tag_name", "v0.7.3"),
        ("filename", "sie_audio_prep-0.7.3-cp312-abi3-manylinux_2_28_x86_64.whl"),
        ("tag_version", "0.7.3"),
        ("tag_filename", "wrong.whl"),
        ("tag_is_ancestor", False),
        ("release_is_stable", False),
    ):
        assert not contract.trusted_audio_repair_context(**{**base, key: value})


@pytest.mark.parametrize("event_name", ["pull_request", "workflow_dispatch"])
def test_pr_and_manual_callers_cannot_authorize_publication(event_name: str) -> None:
    sha = "a" * 40
    assert not contract.trusted_publication_context(
        publish=True,
        latch="true",
        event_name=event_name,
        ref="refs/heads/main",
        ref_protected=True,
        repository="superlinked/sie",
        github_sha=sha,
        input_sha=sha,
        tag_name="v0.7.2",
        version="0.7.2",
        tag_sha=sha,
    )


def test_only_exact_push_main_sha_and_tag_context_authorizes_publication() -> None:
    sha = "a" * 40
    base = {
        "publish": True,
        "latch": "true",
        "event_name": "push",
        "ref": "refs/heads/main",
        "ref_protected": True,
        "repository": "superlinked/sie",
        "github_sha": sha,
        "input_sha": sha,
        "tag_name": "v0.7.2",
        "version": "0.7.2",
        "tag_sha": sha,
    }
    assert contract.trusted_publication_context(**base)
    for key, value in (
        ("ref", "refs/heads/release"),
        ("ref_protected", False),
        ("repository", "someone/fork"),
        ("input_sha", "b" * 40),
        ("tag_sha", "b" * 40),
        ("tag_name", "v0.7.3"),
        ("latch", "false"),
    ):
        untrusted = {**base, key: value}
        assert not contract.trusted_publication_context(**untrusted)
