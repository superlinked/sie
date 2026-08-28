#!/usr/bin/env python3
"""Fail-open path classification for the public CI workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

GROUPS = ("python", "typescript", "rust", "contracts", "helm")
GLOBAL_NAMES = {
    ".github/workflows/ci.yml",
    "mise.toml",
    "package.json",
    "pnpm-lock.yaml",
    "pnpm-workspace.yaml",
    "pyproject.toml",
    "uv.lock",
    "Cargo.toml",
    "Cargo.lock",
    "tools/init.sh",
    "tools/mise_tasks/full-sync.bash",
}
DOC_SUFFIXES = {".md", ".rst", ".txt", ".png", ".jpg", ".jpeg", ".gif", ".svg"}


def all_groups() -> dict[str, bool]:
    return dict.fromkeys(GROUPS, True)


def is_documentation(path: str) -> bool:
    item = Path(path)
    return item.suffix.lower() in DOC_SUFFIXES or path.startswith("docs/")


def classify(paths: list[str], *, event: str) -> dict[str, bool]:
    """Classify a PR diff; pushes and unsafe/unknown paths run everything."""
    if event != "pull_request":
        return all_groups()
    selected = dict.fromkeys(GROUPS, False)
    for path in paths:
        if not path or path in GLOBAL_NAMES or path.startswith((".github/", "tools/mise_tasks/")):
            return all_groups()
        if is_documentation(path):
            continue

        matched = False
        if path.endswith((".py", ".pyi")) or path.startswith(
            ("packages/sie_server/", "packages/sie_sdk/", "integrations/sie_")
        ):
            selected["python"] = True
            matched = True
        if path.endswith((".ts", ".tsx", ".js", ".mjs", ".cjs")) or path.startswith(
            ("packages/sie_ts_sdk/", "integrations/sie_ts_")
        ):
            selected["typescript"] = True
            matched = True
        if path.endswith(".rs") or path.startswith(
            (
                "packages/sie_gateway/",
                "packages/sie_server_sidecar/",
                "packages/sie_server_rust/",
                "packages/sie_telemetry/",
            )
        ):
            selected["rust"] = True
            matched = True
        if (
            path.startswith(("tests/parity/", "conformance/", "packages/wire-fixtures/"))
            or "ipc_types" in path
            or "response_chunk" in path
        ):
            selected["contracts"] = True
            matched = True
        if path.startswith(("deploy/helm/", "packages/sie_server/models/", "packages/sie_server/bundles/")):
            selected["helm"] = True
            matched = True
        if path.startswith("tools/ci/"):
            return all_groups()
        if not matched:
            return all_groups()
    return selected


def changed_paths(base: str, head: str) -> list[str] | None:
    if not base or not head:
        return None
    result = subprocess.run(  # noqa: S603
        ["git", "diff", "--name-only", "-z", "--diff-filter=ACDMRT", base, head, "--"],  # noqa: S607
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    try:
        return [item.decode() for item in result.stdout.split(b"\0") if item]
    except UnicodeDecodeError:
        return None


def write_outputs(selected: dict[str, bool], output_path: str | None) -> None:
    rendered = "".join(f"{group}={'true' if selected[group] else 'false'}\n" for group in GROUPS)
    if output_path:
        with Path(output_path).open("a", encoding="utf-8") as output:
            output.write(rendered)
    else:
        print(rendered, end="")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", required=True)
    parser.add_argument("--base", default="")
    parser.add_argument("--head", default="")
    parser.add_argument("--github-output")
    parser.add_argument("--stdin-nul", action="store_true")
    args = parser.parse_args()

    if args.stdin_nul:
        try:
            paths = [item.decode() for item in sys.stdin.buffer.read().split(b"\0") if item]
        except UnicodeDecodeError:
            paths = None
    elif args.event == "pull_request":
        paths = changed_paths(args.base, args.head)
    else:
        paths = []

    selected = all_groups() if paths is None else classify(paths, event=args.event)
    write_outputs(selected, args.github_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
