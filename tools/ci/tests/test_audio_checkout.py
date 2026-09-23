from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tools.ci import check_release_contract as contract


def audio_workflow() -> str:
    return (contract.ROOT / ".github/workflows/release-audio.yml").read_text()


def audio_build() -> str:
    return contract.workflow_job_blocks(".github/workflows/release-audio.yml")["build"]


def test_audio_checkout_contract() -> None:
    assert contract.audio_checkout_errors(audio_build(), audio_workflow()) == []


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
    assert contract.audio_checkout_errors(build.replace(contract.AUDIO_SOURCE_ASSERTION, replacement), audio_workflow())


@pytest.mark.parametrize(
    "setting", ["if: false", "continue-on-error: true", "shell: bash {0}", "shell: sh", "shell: bash"]
)
@pytest.mark.parametrize("first_key", [False, True])
def test_audio_checkout_rejects_skipped_suppressed_or_shell_overridden_step(setting: str, first_key: bool) -> None:
    build = audio_build()
    original = "      - id: contract\n"
    assert build.count(original) == 1
    replacement = f"      - {setting}\n        id: contract\n" if first_key else f"{original}        {setting}\n"
    assert contract.audio_checkout_errors(build.replace(original, replacement), audio_workflow())


@pytest.mark.parametrize(
    ("workflow_shell", "job_shell", "valid"),
    [
        ("bash", None, True),
        (None, None, False),
        ("sh", None, False),
        ("bash {0}", None, False),
        ("bash", "bash", True),
        ("bash", "sh", False),
        ("bash", "bash {0}", False),
        ("bash", "", False),
        ("sh", "bash", True),
        (None, "bash", True),
    ],
)
def test_audio_checkout_requires_effective_bash_default(workflow_shell, job_shell, valid) -> None:
    workflow = audio_workflow()
    default = "defaults:\n  run:\n    shell: bash\n"
    assert workflow.count(default) == 1
    replacement = f"defaults:\n  run:\n    shell: {workflow_shell}\n" if workflow_shell is not None else ""
    workflow = workflow.replace(default, replacement)
    build = audio_build()
    if job_shell is not None:
        build = build.replace("    steps:\n", f"    defaults:\n      run:\n        shell: {job_shell}\n    steps:\n")
    assert (contract.audio_checkout_errors(build, workflow) == []) is valid


def test_audio_checkout_job_working_directory_default_keeps_workflow_shell() -> None:
    build = audio_build().replace(
        "    steps:\n", "    defaults:\n      run:\n        working-directory: .\n    steps:\n"
    )
    assert contract.audio_checkout_errors(build, audio_workflow()) == []


@pytest.mark.parametrize("scope", ["workflow", "job"])
@pytest.mark.parametrize("shell", ["bash", "sh"])
@pytest.mark.parametrize("comment_after", ["defaults", "run"])
def test_audio_checkout_default_comments_do_not_change_shell_precedence(scope, shell, comment_after) -> None:
    workflow = audio_workflow()
    build = audio_build()
    indent = "" if scope == "workflow" else "    "
    defaults = f"{indent}defaults:\n{indent}  run:\n{indent}    shell: {shell}\n"
    defaults = defaults.replace(f"{comment_after}:\n", f"{comment_after}:\n# unindented comment\n")
    if scope == "workflow":
        workflow = workflow.replace("defaults:\n  run:\n    shell: bash\n", defaults)
    else:
        build = build.replace("    steps:\n", f"{defaults}    steps:\n")
    assert (contract.audio_checkout_errors(build, workflow) == []) is (shell == "bash")


@pytest.mark.parametrize("setting", ["if: false", "continue-on-error: true"])
def test_audio_checkout_rejects_skipped_or_suppressed_job(setting: str) -> None:
    build = audio_build().replace("  build:\n", f"  build:\n    {setting}\n")
    assert contract.audio_checkout_errors(build, audio_workflow())


@pytest.mark.parametrize(
    "defaults",
    [
        "defaults: {run: {shell: bash}}\n",
        "defaults:\n  run:\n    shell: bash\n    shell: sh\n",
        "defaults:\n  run: {}\n  shell: bash\n",
    ],
)
def test_audio_checkout_rejects_unsupported_or_ambiguous_defaults(defaults: str) -> None:
    workflow = audio_workflow().replace("defaults:\n  run:\n    shell: bash\n", defaults)
    assert contract.audio_checkout_errors(audio_build(), workflow)


@pytest.mark.parametrize("default", ["", "defaults:\n  run:\n    shell: sh\n"])
def test_audio_release_contract_validates_workflow_shell_context(monkeypatch, default: str) -> None:
    workflow_path = contract.ROOT / ".github/workflows/release-audio.yml"
    read_text = Path.read_text
    changed = audio_workflow().replace("defaults:\n  run:\n    shell: bash\n", default)
    monkeypatch.setattr(Path, "read_text", lambda path: changed if path == workflow_path else read_text(path))
    assert (
        "native audio source check must compare the exact SHA with command-scoped workspace Git trust"
        in contract.audio_release_errors()
    )


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
