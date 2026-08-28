from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.mise_tasks import docker_task

FULL_SHA = "a" * 40


def test_release_matrix_resolves_current_supported_pairs() -> None:
    targets = docker_task.load_release_matrix()
    assert len(targets) == 7
    assert {(target.platform, target.bundle) for target in targets} == {
        (platform, bundle) for platform in ("cuda12", "cpu") for bundle in ("default", "sglang", "transformers5")
    } | {("cuda13", "sglang-cu130")}
    assert targets[-1].target == "gemma"


def test_release_matrix_rejects_duplicates(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.json"
    matrix.write_text(
        json.dumps(
            {
                "platforms": ["cuda12"],
                "bundles": ["default"],
                "include": [{"platform": "cuda12", "bundle": "default"}],
            }
        )
    )
    with pytest.raises(ValueError, match="duplicate"):
        docker_task.load_release_matrix(matrix)


def test_release_matrix_rejects_bundle_platform_disagreement(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.json"
    matrix.write_text(
        json.dumps(
            {
                "platforms": [],
                "bundles": [],
                "include": [{"platform": "cuda13", "bundle": "default"}],
            }
        )
    )
    with pytest.raises(ValueError, match="disagrees"):
        docker_task.load_release_matrix(matrix)


def test_server_build_uses_full_source_revision_and_versioned_tag() -> None:
    target = docker_task.ServerTarget("cpu", "default")
    command = docker_task.build_server_command(
        registry="ghcr.io/superlinked",
        version="0.7.2",
        target=target,
        source_revision=FULL_SHA,
        push=True,
    )
    assert f"SIE_SRC_REV={FULL_SHA}" in command
    assert "ghcr.io/superlinked/sie-server:0.7.2-cpu-default" in command
    assert "--push" in command
    with pytest.raises(ValueError, match="full 40-character"):
        docker_task.build_server_command(
            registry="ghcr.io/superlinked",
            version="0.7.2",
            target=target,
            source_revision="abc123",
            push=False,
        )


def test_complete_set_is_verified_before_alias_commands(monkeypatch) -> None:
    targets = docker_task.load_release_matrix()
    commands: list[list[str]] = []

    def fail_verification(*_args) -> None:
        raise RuntimeError("incomplete")

    monkeypatch.setattr(docker_task, "verify_release", fail_verification)
    monkeypatch.setattr(docker_task, "run", commands.append)
    with pytest.raises(RuntimeError, match="incomplete"):
        docker_task.move_aliases("ghcr.io/superlinked", "0.7.2", targets)
    assert commands == []


def test_expected_release_set_has_no_duplicates() -> None:
    targets = docker_task.load_release_matrix()
    images = docker_task.expected_versioned_images("ghcr.io/superlinked", "0.7.2", targets)
    assert len(images) == 12
    assert len(images) == len(set(images))
    assert "ghcr.io/superlinked/sie-server-rust:0.7.2-cuda12-sm89" in images
