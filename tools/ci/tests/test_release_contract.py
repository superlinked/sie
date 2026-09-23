from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import textwrap
import tomllib
from copy import deepcopy
from pathlib import Path

import pytest

from tools.ci import check_release_contract as contract
from tools.ci import release_openapi


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


def test_release_please_bootstrap_is_the_exact_public_v073_commit() -> None:
    config = contract.load_json("release-please-config.json")
    assert config["bootstrap-sha"] == contract.RELEASE_BOOTSTRAP_SHA
    assert config["bootstrap-sha"] == "60996d9c30168e0f8e85b680295f147fdee87f61"
    assert "bootstrap-sha" not in config["packages"]["."]


@pytest.mark.parametrize("bootstrap_sha", [None, "60996d9", "b" * 40])
def test_release_please_rejects_any_other_bootstrap_boundary(monkeypatch, bootstrap_sha) -> None:
    real_load_json = contract.load_json

    def load_json(path):
        document = real_load_json(path)
        if path == "release-please-config.json":
            document = deepcopy(document)
            if bootstrap_sha is None:
                document.pop("bootstrap-sha")
            else:
                document["bootstrap-sha"] = bootstrap_sha
        return document

    monkeypatch.setattr(contract, "load_json", load_json)
    assert "release-please bootstrap-sha must be the exact public v0.7.3 commit" in contract.release_config_errors()


def openapi_surfaces() -> tuple[str, str, set[str]]:
    refresh = contract.workflow_job_blocks(".github/workflows/release.yml")["release-please"]
    contracts = contract.workflow_job_blocks(".github/workflows/ci.yml")["contracts"]
    return refresh, contracts, set(release_openapi.OPENAPI_VERSION_SOURCES)


@pytest.mark.parametrize("document", sorted(contract.OPENAPI_VERSION_PATHS))
def test_release_pr_stamps_every_openapi_version_ci_regenerates(document) -> None:
    refresh, contracts, stamped = openapi_surfaces()
    assert contract.release_openapi_errors(refresh, contracts, stamped) == []

    assert contract.release_openapi_errors(refresh, contracts, stamped - {document})
    assert contract.release_openapi_errors(refresh, contracts.replace(document, ""), stamped)
    for command in ("git diff --quiet --", "git add"):
        unstaged = re.sub(rf"({re.escape(command)} .*) {re.escape(document)}", r"\1", refresh)
        assert unstaged != refresh
        assert contract.release_openapi_errors(unstaged, contracts, stamped)
    unstamped = refresh.replace(contract.OPENAPI_STAMP_COMMAND, "")
    assert contract.release_openapi_errors(unstamped, contracts, stamped)


@pytest.mark.parametrize(
    ("anchor", "offset", "valid"),
    [
        ('git checkout -B "$branch" FETCH_HEAD', 1, True),
        ("mise exec -- cargo metadata", 1, True),
        ('git checkout -B "$branch" FETCH_HEAD', 0, False),
        ("git diff --quiet --", 1, False),
        ("git add ", 1, False),
    ],
)
def test_release_pr_stamps_openapi_versions_after_checkout_and_before_commit(anchor, offset, valid) -> None:
    refresh, contracts, stamped = openapi_surfaces()
    lines = refresh.splitlines()
    stamp = next(line for line in lines if line.strip() == contract.OPENAPI_STAMP_COMMAND)
    lines.remove(stamp)
    lines.insert(next(index for index, line in enumerate(lines) if anchor in line) + offset, stamp)
    assert (contract.release_openapi_errors("\n".join(lines), contracts, stamped) == []) is valid


@pytest.mark.parametrize(
    "stamp",
    [
        f"{contract.OPENAPI_STAMP_COMMAND} || true",
        f"# {contract.OPENAPI_STAMP_COMMAND}",
        contract.OPENAPI_STAMP_COMMAND.replace(" -I ", " "),
    ],
)
def test_release_pr_stamp_fails_closed_in_isolated_python(stamp) -> None:
    refresh, contracts, stamped = openapi_surfaces()
    weakened = refresh.replace(contract.OPENAPI_STAMP_COMMAND, stamp)
    assert weakened != refresh
    assert contract.release_openapi_errors(weakened, contracts, stamped)


def line_index(lines: list[str], text: str) -> int:
    return next(index for index, line in enumerate(lines) if text in line)


def duplicate_diff_check(lines: list[str]) -> None:
    start = line_index(lines, "git diff --quiet --")
    lines[start:start] = lines[start : start + 3]


def duplicate_staging(lines: list[str]) -> None:
    stage = line_index(lines, "git add ")
    lines.insert(stage, lines[stage])


def move_before(lines: list[str], moved: str, anchor: str) -> None:
    line = lines.pop(line_index(lines, moved))
    lines.insert(line_index(lines, anchor), line)


@pytest.mark.parametrize(
    ("command", "replacement"),
    [
        (contract.SDK_INSTALL_COMMAND, ""),
        (contract.SDK_INSTALL_COMMAND, f"{contract.SDK_INSTALL_COMMAND} || true"),
        (
            contract.SDK_INSTALL_COMMAND,
            contract.SDK_INSTALL_COMMAND.replace("--frozen-lockfile", "--no-frozen-lockfile"),
        ),
        (contract.SDK_INSTALL_COMMAND, contract.SDK_INSTALL_COMMAND.replace(" --ignore-scripts", "")),
        (contract.SDK_INSTALL_COMMAND, contract.SDK_INSTALL_COMMAND.replace("@superlinked/sie-sdk", "other-package")),
        (contract.SDK_FORMAT_COMMAND, ""),
        (contract.SDK_FORMAT_COMMAND, f"# {contract.SDK_FORMAT_COMMAND}"),
        (contract.SDK_FORMAT_COMMAND, f"echo {contract.SDK_FORMAT_COMMAND}"),
        (contract.SDK_FORMAT_COMMAND, f"{contract.SDK_FORMAT_COMMAND} || true"),
        (
            contract.SDK_FORMAT_COMMAND,
            "mise exec -- pnpm dlx @biomejs/biome format --write packages/sie_ts_sdk/package.json",
        ),
        (contract.SDK_FORMAT_COMMAND, contract.SDK_FORMAT_COMMAND.replace(" --write", "")),
        (
            contract.SDK_FORMAT_COMMAND,
            contract.SDK_FORMAT_COMMAND.replace("packages/sie_ts_sdk", "integrations/sie_ts_chroma"),
        ),
        (contract.SDK_FORMAT_COMMAND, contract.SDK_FORMAT_COMMAND.replace("package.json", "other.json")),
    ],
    ids=[
        "missing-install",
        "suppressed-install",
        "unfrozen-install",
        "install-scripts",
        "wrong-package-install",
        "missing-formatter",
        "commented-formatter",
        "echoed-formatter",
        "suppressed-formatter",
        "external-formatter",
        "read-only-formatter",
        "wrong-package-formatter",
        "wrong-file-formatter",
    ],
)
def test_release_pr_metadata_refresh_uses_active_exact_pinned_commands(command, replacement) -> None:
    refresh, contracts, stamped = openapi_surfaces()
    assert refresh.count(command) == 1
    assert contract.release_openapi_errors(refresh.replace(command, replacement), contracts, stamped)


@pytest.mark.parametrize(
    ("moved", "anchor"),
    [
        (contract.SDK_INSTALL_COMMAND, "mise exec -- pnpm install --lockfile-only"),
        (contract.SDK_FORMAT_COMMAND, contract.SDK_INSTALL_COMMAND),
        (contract.SDK_FORMAT_COMMAND, "git add "),
        (contract.SDK_FORMAT_COMMAND, "git commit "),
    ],
    ids=["install-before-lock", "format-before-install", "diff-before-format", "add-before-format"],
)
def test_release_pr_metadata_refresh_preserves_lock_install_format_diff_stage_order(moved, anchor) -> None:
    refresh, contracts, stamped = openapi_surfaces()
    lines = refresh.splitlines()
    move_before(lines, moved, anchor)
    assert contract.release_openapi_errors("\n".join(lines), contracts, stamped)


@pytest.mark.parametrize("command", ["if git diff --quiet --", "git add "])
def test_release_pr_metadata_changes_are_detected_and_staged(command) -> None:
    refresh, contracts, stamped = openapi_surfaces()
    lines = refresh.splitlines()
    index = line_index(lines, command)
    assert contract.SDK_PACKAGE_PATH in lines[index]
    lines[index] = lines[index].replace(f" {contract.SDK_PACKAGE_PATH}", "")
    assert contract.release_openapi_errors("\n".join(lines), contracts, stamped)


def test_release_pr_metadata_refresh_cannot_move_to_a_skipped_step() -> None:
    refresh, contracts, stamped = openapi_surfaces()
    lines = refresh.splitlines()
    index = line_index(lines, contract.SDK_FORMAT_COMMAND)
    lines[index:index] = ["      - name: Skip formatting", "        if: false", "        run: |"]
    assert contract.release_openapi_errors("\n".join(lines), contracts, stamped)


@pytest.mark.parametrize(
    "rewrite",
    [
        duplicate_diff_check,
        duplicate_staging,
        lambda lines: move_before(lines, "git add ", "git diff --quiet --"),
        lambda lines: move_before(lines, "git commit ", "git add "),
        lambda lines: move_before(lines, "git push origin ", "git commit "),
    ],
    ids=["duplicate-diff", "duplicate-add", "add-before-diff", "commit-before-add", "push-before-commit"],
)
def test_release_pr_refresh_diffs_adds_commits_and_pushes_once_in_order(rewrite) -> None:
    refresh, contracts, stamped = openapi_surfaces()
    lines = refresh.splitlines()
    rewrite(lines)
    assert lines != refresh.splitlines()
    assert contract.release_openapi_errors("\n".join(lines), contracts, stamped)


REFRESH_RUN = "        run: |\n          set -euo pipefail\n          branch="


@pytest.mark.parametrize(
    ("surface", "old", "new"),
    [
        ("contracts", "- run: mise run openapi\n", "# - run: mise run openapi\n"),
        ("contracts", "- run: mise run openapi\n", "- run: echo mise run openapi\n"),
        ("contracts", "- run: mise run openapi\n", "- run: mise run openapi\n        if: false\n"),
        ("contracts", "packages/sie_gateway/openapi.json\n", "packages/sie_gateway/openapi.json\n          || true\n"),
        (
            "contracts",
            "packages/sie_gateway/openapi.json\n",
            "packages/sie_gateway/openapi.json\n        continue-on-error: true\n",
        ),
        ("contracts", "    timeout-minutes: 25\n", "    timeout-minutes: 25\n    continue-on-error: true\n"),
        ("refresh", "if git diff --quiet --", "if echo git diff --quiet --"),
        ("refresh", "git add uv.lock", "git add --dry-run uv.lock"),
        ("refresh", "packages/sie_gateway/openapi.json\n", "packages/sie_gateway/openapi.json || true\n"),
        ("refresh", "OpenAPI versions'\n", "OpenAPI versions' || true\n"),
        ("refresh", '"HEAD:refs/heads/$branch"\n', '"HEAD:refs/heads/$branch" || true\n'),
        ("refresh", f"{contract.OPENAPI_STAMP_COMMAND}\n", f"set +e\n          {contract.OPENAPI_STAMP_COMMAND}\n"),
        ("refresh", f"{contract.OPENAPI_STAMP_COMMAND}\n", f"exit 0\n          {contract.OPENAPI_STAMP_COMMAND}\n"),
        ("refresh", REFRESH_RUN, f"        continue-on-error: true\n{REFRESH_RUN}"),
        ("refresh", REFRESH_RUN, f"        shell: bash {{0}}\n{REFRESH_RUN}"),
    ],
    ids=[
        "commented-regeneration",
        "echoed-regeneration",
        "skipped-regeneration",
        "diff-or-true",
        "diff-continue-on-error",
        "contracts-job-continue-on-error",
        "echoed-refresh-diff",
        "dry-run-add",
        "add-or-true",
        "commit-or-true",
        "push-or-true",
        "set-plus-e",
        "early-exit",
        "refresh-continue-on-error",
        "refresh-shell-without-errexit",
    ],
)
def test_release_openapi_contract_rejects_inactive_or_suppressed_commands(surface, old, new) -> None:
    refresh, contracts, stamped = openapi_surfaces()
    surfaces = {"refresh": refresh, "contracts": contracts}
    assert surfaces[surface].count(old) == 1
    surfaces[surface] = surfaces[surface].replace(old, new)
    assert contract.release_openapi_errors(surfaces["refresh"], surfaces["contracts"], stamped)


@pytest.mark.parametrize(
    ("opened", "closed"),
    [("if false; then", "fi"), ("while false; do", "done")],
    ids=["if-false", "while-false"],
)
def test_release_pr_refresh_rejects_inactive_shell_wrappers(opened, closed) -> None:
    refresh, contracts, stamped = openapi_surfaces()
    opening = f"        run: |\n          set -euo pipefail\n          {opened}\n          branch="
    wrapped = refresh.replace(REFRESH_RUN, opening).replace(
        '"HEAD:refs/heads/$branch"\n', f'"HEAD:refs/heads/$branch"\n          {closed}\n'
    )
    assert wrapped != refresh
    assert contract.release_openapi_errors(wrapped, contracts, stamped)


def test_release_pr_refresh_rejects_commands_moved_into_another_step() -> None:
    refresh, contracts, stamped = openapi_surfaces()
    lines = refresh.splitlines()
    lines.insert(line_index(lines, 'pr_number="$(jq'), lines.pop(line_index(lines, "git push origin")))
    moved = "\n".join(lines)
    assert moved != refresh
    assert contract.release_openapi_errors(moved, contracts, stamped)


@pytest.mark.parametrize("document", sorted(contract.OPENAPI_VERSION_PATHS))
def test_release_please_must_not_rewrite_generated_openapi(monkeypatch, document) -> None:
    real_load_json = contract.load_json

    def load_json(path):
        loaded = real_load_json(path)
        if path == "release-please-config.json":
            loaded = deepcopy(loaded)
            extra_file = {"type": "json", "path": document, "jsonpath": "$.info.version"}
            loaded["packages"]["."]["extra-files"].append(extra_file)
        return loaded

    monkeypatch.setattr(contract, "load_json", load_json)
    monkeypatch.setattr(contract, "EXTRA_VERSION_PATHS", contract.EXTRA_VERSION_PATHS | {document})
    assert contract.release_config_errors() == [
        "generated OpenAPI documents must be stamped, not rewritten by release-please"
    ]


def test_option_ext_mpl_exception_is_exact_and_cannot_broaden() -> None:
    policy = tomllib.loads((contract.ROOT / "deny.toml").read_text())
    assert contract._license_policy_errors(policy) == []

    globally_allowed = deepcopy(policy)
    globally_allowed["licenses"]["allow"].append("MPL-2.0")
    assert "MPL-2.0 must not be globally allowed" in contract._license_policy_errors(globally_allowed)

    wildcard_version = deepcopy(policy)
    option_ext = next(entry for entry in wildcard_version["licenses"]["exceptions"] if entry["name"] == "option-ext")
    option_ext["version"] = "*"
    assert "option-ext MPL-2.0 allowance must be confined to exact version 0.2.0" in contract._license_policy_errors(
        wildcard_version
    )

    extra_crate = deepcopy(policy)
    extra_crate["licenses"]["exceptions"].append({"name": "unreviewed", "allow": ["MPL-2.0"]})
    assert (
        "cargo-deny MPL-2.0 exception surface differs from the reviewed crate set"
        in contract._license_policy_errors(extra_crate)
    )


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
        for family in (
            "prepare",
            "artifacts-ready",
            "python-publish",
            "npm-publish",
            "docker",
            "helm",
            "audio",
            "native",
        )
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


def test_helm_release_must_package_the_staged_and_validated_catalog(monkeypatch) -> None:
    read_text = Path.read_text

    def unstaged_package(path, *args, **kwargs):
        text = read_text(path, *args, **kwargs)
        if path == contract.ROOT / ".github/workflows/release-helm.yml":
            text = text.replace(
                "mise run helm -- package --destination artifact",
                "mise exec -- helm package deploy/helm/sie-cluster --destination artifact",
            )
        return text

    monkeypatch.setattr(Path, "read_text", unstaged_package)
    assert any("mise run helm -- package" in error for error in contract.helm_release_errors())


def test_public_release_app_hands_final_pr_head_to_ci() -> None:
    assert contract.release_app_errors() == []


def test_release_authoring_is_default_off_until_its_app_is_configured() -> None:
    block = contract.workflow_job_blocks(".github/workflows/release.yml")["release-please"]
    condition = contract.job_scalar(block, "if")
    assert condition is not None
    assert "vars.PUBLIC_RELEASE_AUTOMATION_ENABLED == 'true'" in condition
    assert "vars.PUBLIC_RELEASE_AUTOMATION_ENABLED != 'false'" not in condition
    assert "PUBLIC_RELEASE_PUBLISHING_ENABLED" not in condition
    assert contract.release_automation_gate_errors(condition) == []

    gate = "vars.PUBLIC_RELEASE_AUTOMATION_ENABLED == 'true'"
    for unsafe in (
        condition.replace(gate, "vars.PUBLIC_RELEASE_AUTOMATION_ENABLED != 'false'"),
        f"{condition} || true",
        condition.replace(gate, f"({gate} || vars.PUBLIC_RELEASE_AUTOMATION_ENABLED == '')"),
    ):
        assert contract.release_automation_gate_errors(unsafe)


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
    assert "mise --no-config install python@3.12.12 uv@0.5.31 zig@0.13.0 rust@1.98.1" in workflow
    assert "rust@1.98.1 -- rustc --version" in workflow
    assert "rust@1.98.1 -- cargo --version" in workflow
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
