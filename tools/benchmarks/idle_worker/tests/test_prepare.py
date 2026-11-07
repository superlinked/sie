"""Snapshot provenance must stay stable even if an input branch moves."""

import json
import sys
from pathlib import Path

import pytest

from tools.benchmarks.idle_worker import prepare


def test_diff_describes_snapshots_when_input_branch_moves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    monkeypatch.setattr(prepare, "INPUTS", tmp_path / "_inputs")
    git = prepare.git
    git("init", "--quiet")
    (tmp_path / "uv.lock").write_text("same lock\n")
    for directory in prepare.PATHS:
        target = tmp_path / directory
        target.mkdir(parents=True)
        (target / "probe.txt").write_text("baseline\n")

    def commit() -> str:
        git("add", ".")
        git(
            "-c",
            "user.name=Benchmark Test",
            "-c",
            "user.email=benchmark@example.test",
            "commit",
            "--quiet",
            "-m",
            "snapshot",
        )
        return git("rev-parse", "HEAD").decode().strip()

    baseline = commit()
    (tmp_path / prepare.PATHS[0] / "probe.txt").write_text("candidate\n")
    candidate = commit()
    git("branch", "baseline", baseline)
    git("branch", "candidate", candidate)
    expected_diff = git("diff", "--stat", baseline, candidate, "--", *prepare.PATHS).decode()
    (tmp_path / prepare.PATHS[0] / "later.txt").write_text("not in the archived candidate\n")
    moved_candidate = commit()

    def git_with_moving_branch(*args: str) -> bytes:
        if args[0] == "diff":
            git("update-ref", "refs/heads/candidate", moved_candidate)
        return git(*args)

    monkeypatch.setattr(prepare, "git", git_with_moving_branch)
    monkeypatch.setattr(sys, "argv", ["prepare.py", "--baseline", "baseline", "--candidate", "candidate"])
    prepare.main()

    manifest = json.loads((prepare.INPUTS / "manifest.json").read_text())
    assert manifest["baseline"]["commit"] == baseline
    assert manifest["candidate"]["commit"] == candidate
    assert git("rev-parse", "candidate").decode().strip() == moved_candidate
    assert manifest["source_diff"] == expected_diff
