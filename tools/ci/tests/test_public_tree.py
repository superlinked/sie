from __future__ import annotations

import re
from pathlib import Path

from tools.ci import check_public_tree

REPOSITORY_ROOT = check_public_tree.REPOSITORY_ROOT


def test_reference_guard_reports_forbidden_text(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(check_public_tree, "REPOSITORY_ROOT", tmp_path)
    clean = tmp_path / "clean.txt"
    clean.write_text("https://github.com/superlinked/sie\n", encoding="utf-8")
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"packages/" + b"sie_cloud" + b"/gateway\n")
    findings = check_public_tree.violations([clean, bad])
    assert len(findings) == 1
    assert "forbidden public-tree reference" in findings[0]


def test_bootstrap_uses_root_locks_even_in_ci() -> None:
    full_sync = (REPOSITORY_ROOT / "tools/mise_tasks/full-sync.bash").read_text()
    init = (REPOSITORY_ROOT / "tools/init.sh").read_text()
    package = (REPOSITORY_ROOT / "package.json").read_text()
    mise_config = (REPOSITORY_ROOT / "mise.toml").read_text()
    assert "mise run full-sync" in init
    assert "mise run sync" in full_sync
    assert "pnpm install --frozen-lockfile" in full_sync
    assert "packages/sie_ts_sdk" not in full_sync
    assert "CI:-" not in full_sync
    assert '"packageManager": "pnpm@9.15.9"' in package
    assert '"prepare": "pnpm run build"' in (REPOSITORY_ROOT / "packages/sie_ts_sdk/package.json").read_text()
    assert "MISE_" not in mise_config
    assert "XDG_CONFIG_HOME" not in mise_config
    assert "XDG_STATE_HOME" not in mise_config
    assert not (REPOSITORY_ROOT / ".npmrc").exists()
    assert not (REPOSITORY_ROOT / "packages/sie_ts_sdk/pnpm-lock.yaml").exists()


def test_ci_is_fork_safe_and_actions_are_immutable() -> None:
    workflow = (REPOSITORY_ROOT / ".github/workflows/ci.yml").read_text()
    assert "pull_request_target" not in workflow
    assert "secrets." not in workflow
    assert "self-hosted" not in workflow
    assert "permissions:\n  contents: read" in workflow
    for name in (
        "CI / Policy",
        "CI / Python",
        "CI / TypeScript",
        "CI / Rust",
        "CI / Contracts",
        "CI / Helm",
        "CI / Required",
    ):
        assert f"name: {name}" in workflow
    action_lines = [line.strip() for line in workflow.splitlines() if "uses:" in line]
    assert action_lines
    assert all(re.search(r"@[0-9a-f]{40}(?:\s|$)", line) for line in action_lines)
