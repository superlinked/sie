#!/usr/bin/env python3
"""Reject references that cannot be resolved from the public repository."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = (
    b"superlinked/" + b"sie-" + b"internal",
    b"sie-" + b"internal#",
    b"packages/" + b"sie_admin",
    b"packages/" + b"sie_bench",
    b"packages/" + b"sie_tools",
    b"packages/" + b"sie_cloud",
    b"tools/" + b"internal_python",
)
ARCHIVE_GENERATED_DIRS = {
    ".cache",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
}


def candidate_paths() -> list[Path]:
    """Return tracked and non-ignored untracked files for local and CI checks."""
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],  # noqa: S607
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode == 0:
        return [REPOSITORY_ROOT / item.decode() for item in result.stdout.split(b"\0") if item]
    return sorted(
        path
        for path in REPOSITORY_ROOT.rglob("*")
        if path.is_file()
        and not ARCHIVE_GENERATED_DIRS.intersection(path.relative_to(REPOSITORY_ROOT).parts)
        and "deploy/helm/sie-cluster/charts" not in path.as_posix()
    )


def violations(paths: list[Path]) -> list[str]:
    findings: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        for line_number, line in enumerate(data.splitlines(), start=1):
            for forbidden in FORBIDDEN:
                if forbidden in line:
                    findings.append(
                        f"{path.relative_to(REPOSITORY_ROOT)}:{line_number}: forbidden public-tree reference"
                    )
                    break
    return findings


def main() -> int:
    findings = violations(candidate_paths())
    if findings:
        print("\n".join(findings))
        return 1
    print("Public-tree reference check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
