from __future__ import annotations

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


def test_public_seed_tag_is_ancestral() -> None:
    assert contract.tag_errors() == []


def test_candle_and_docker_release_source_closure() -> None:
    assert contract.candle_source_errors() == []
    assert contract.docker_copy_errors() == []
    assert contract.docker_release_errors() == []


def test_helm_release_follows_verified_images() -> None:
    assert contract.helm_release_errors() == []
