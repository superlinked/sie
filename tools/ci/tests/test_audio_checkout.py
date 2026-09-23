from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tools.ci import check_release_contract as contract


def audio_build() -> str:
    return contract.workflow_job_blocks(".github/workflows/release-audio.yml")["build"]


def test_audio_checkout_contract() -> None:
    assert contract.audio_checkout_errors(audio_build()) == []


@pytest.mark.parametrize(
    "replacement",
    [
        "",
        f"# {contract.AUDIO_SOURCE_ASSERTION}",
        f"echo {contract.AUDIO_SOURCE_ASSERTION}",
        f"{contract.AUDIO_SOURCE_ASSERTION} || true",
        contract.AUDIO_SOURCE_ASSERTION.replace(' -c safe.directory="$GITHUB_WORKSPACE"', ""),
        contract.AUDIO_SOURCE_ASSERTION.replace("$GITHUB_WORKSPACE", "*"),
        contract.AUDIO_SOURCE_ASSERTION.replace("$RELEASE_SHA", "$OTHER_SHA"),
        'git config --global --add safe.directory "$GITHUB_WORKSPACE"\n          ' + contract.AUDIO_SOURCE_ASSERTION,
        f"set +e\n          {contract.AUDIO_SOURCE_ASSERTION}",
        f"if false; then\n          {contract.AUDIO_SOURCE_ASSERTION}\n          fi",
    ],
    ids=[
        "missing",
        "commented",
        "echoed",
        "suppressed",
        "missing-trust",
        "wildcard-trust",
        "wrong-source",
        "global-trust",
        "no-errexit",
        "inactive-shell",
    ],
)
def test_audio_checkout_rejects_weakened_source_check(replacement: str) -> None:
    build = audio_build()
    assert build.count(contract.AUDIO_SOURCE_ASSERTION) == 1
    assert contract.audio_checkout_errors(build.replace(contract.AUDIO_SOURCE_ASSERTION, replacement))


@pytest.mark.parametrize("setting", ["if: false", "continue-on-error: true", "shell: bash {0}"])
def test_audio_checkout_rejects_skipped_or_suppressed_step(setting: str) -> None:
    build = audio_build()
    original = "      - id: contract\n"
    assert build.count(original) == 1
    assert contract.audio_checkout_errors(build.replace(original, f"{original}        {setting}\n"))


def test_audio_checkout_ownership_trust_is_command_and_workspace_scoped(tmp_path: Path) -> None:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update({"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull, "LC_ALL": "C"})
    workspace = (tmp_path / "workspace").resolve()
    other = (tmp_path / "other").resolve()
    for repository in (workspace, other):
        repository.mkdir()
        subprocess.run(["/usr/bin/git", "init", "--quiet"], cwd=repository, env=environment, check=True)
        subprocess.run(
            [
                "/usr/bin/git",
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "--quiet",
                "--allow-empty",
                "-m",
                "test",
            ],
            cwd=repository,
            env=environment,
            check=True,
        )
    sha = subprocess.check_output(
        ["/usr/bin/git", "rev-parse", "HEAD"], cwd=workspace, env=environment, text=True
    ).strip()
    environment.update({"GIT_TEST_ASSUME_DIFFERENT_OWNER": "1", "GITHUB_WORKSPACE": str(workspace), "RELEASE_SHA": sha})
    plain = subprocess.run(
        ["/usr/bin/git", "rev-parse", "HEAD"],
        cwd=workspace,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert plain.returncode != 0
    assert "dubious ownership" in plain.stderr

    assertion = next(line.strip() for line in audio_build().splitlines() if "rev-parse" in line)
    for expected_sha, expected_success in ((sha, True), ("0" * 40, False)):
        result = subprocess.run(  # noqa: S603 - execute the repository workflow assertion in an isolated test repo
            ["/bin/bash", "-c", f"set -euo pipefail\n{assertion}"],
            cwd=workspace,
            env={**environment, "RELEASE_SHA": expected_sha},
            capture_output=True,
            text=True,
            check=False,
        )
        assert (result.returncode == 0) is expected_success, result.stderr
        assert "dubious ownership" not in result.stderr

    for repository, command in (
        (workspace, ["/usr/bin/git", "rev-parse", "HEAD"]),
        (other, ["/usr/bin/git", "-c", f"safe.directory={workspace}", "rev-parse", "HEAD"]),
    ):
        result = subprocess.run(  # noqa: S603 - fixed Git commands in isolated test repositories
            command, cwd=repository, env=environment, capture_output=True, text=True, check=False
        )
        assert result.returncode != 0
        assert "dubious ownership" in result.stderr
