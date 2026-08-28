from __future__ import annotations

from pathlib import Path

import pytest

from tools.ci import check_release_contract as contract


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
