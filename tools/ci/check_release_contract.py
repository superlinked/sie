#!/usr/bin/env python3
"""Validate the public version, artifact, and fail-closed publication contract."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PYTHON_DISTRIBUTIONS = (
    "sie-sdk",
    "sie-server",
    "sie-langchain",
    "sie-llamaindex",
    "sie-haystack",
    "sie-dspy",
    "sie-crewai",
    "sie-chroma",
    "sie-lancedb",
    "sie-qdrant",
    "sie-weaviate",
)
NPM_PACKAGES = (
    "@superlinked/sie-sdk",
    "@superlinked/sie-chroma",
    "@superlinked/sie-langchain",
    "@superlinked/sie-llamaindex",
    "@superlinked/sie-lancedb",
)
EXTRA_VERSION_PATHS = {
    "packages/sie_sdk/pyproject.toml",
    "packages/sie_server/pyproject.toml",
    "integrations/sie_langchain/pyproject.toml",
    "integrations/sie_llamaindex/pyproject.toml",
    "integrations/sie_haystack/pyproject.toml",
    "integrations/sie_dspy/pyproject.toml",
    "integrations/sie_crewai/pyproject.toml",
    "integrations/sie_chroma/pyproject.toml",
    "integrations/sie_lancedb/pyproject.toml",
    "integrations/sie_qdrant/pyproject.toml",
    "integrations/sie_weaviate/pyproject.toml",
    "packages/sie_ts_sdk/package.json",
    "integrations/sie_ts_chroma/package.json",
    "integrations/sie_ts_langchain/package.json",
    "integrations/sie_ts_llamaindex/package.json",
    "integrations/sie_ts_lancedb/package.json",
    "packages/sie_ts_sdk/src/version.ts",
    "packages/sie_gateway/Cargo.toml",
    "packages/sie_server_sidecar/Cargo.toml",
    "packages/sie_audio_prep/Cargo.toml",
    "packages/sie_audio_prep/pyproject.toml",
    "packages/sie_audio_prep/build_wheel.py",
    "deploy/helm/sie-cluster/Chart.yaml",
}
ACTION_PIN = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")


def load_json(path: str) -> Any:
    return json.loads((ROOT / path).read_text())


def python_matrices() -> tuple[tuple[str, ...], tuple[str, ...]]:
    workflow = (ROOT / ".github/workflows/release-python.yml").read_text()
    build_section, publish_section = workflow.split("\n  publish:", maxsplit=1)
    build = tuple(re.findall(r"distribution:\s*(sie-[a-z0-9-]+)", build_section))
    publish = tuple(re.findall(r"^\s+- (sie-[a-z0-9-]+)\s*$", publish_section, re.MULTILINE))
    return build, publish


def npm_matrix() -> tuple[str, ...]:
    workflow = (ROOT / ".github/workflows/release-npm.yml").read_text()
    publish_section = workflow.split("\n  publish:", maxsplit=1)[1]
    return tuple(re.findall(r"package:\s*'(@superlinked/sie-[a-z0-9-]+)'", publish_section))


def workflow_pin_errors() -> list[str]:
    errors: list[str] = []
    for path in sorted((ROOT / ".github/workflows").glob("*.yml")):
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            match = re.search(r"\buses:\s*([^\s#]+)", line)
            if not match or match.group(1).startswith("./"):
                continue
            if not ACTION_PIN.fullmatch(match.group(1)):
                errors.append(f"{path.relative_to(ROOT)}:{line_number}: action is not pinned by full SHA")
    return errors


def release_config_errors() -> list[str]:
    errors: list[str] = []
    manifest = load_json(".release-please-manifest.json")
    if manifest != {".": "0.7.2"}:
        errors.append("release-please manifest must contain only the public 0.7.2 seed")
    config = load_json("release-please-config.json")
    packages = config.get("packages", {})
    if set(packages) != {"."} or packages["."].get("release-type") != "simple":
        errors.append("release-please must define one simple root release")
        return errors
    extra_paths = {item["path"] for item in packages["."]["extra-files"]}
    if extra_paths != EXTRA_VERSION_PATHS:
        errors.append("release-please extra-file version surface differs from the public contract")
    for path in extra_paths:
        if not (ROOT / path).is_file():
            errors.append(f"release-please extra file does not exist: {path}")
    return errors


def release_workflow_errors() -> list[str]:
    errors: list[str] = []
    top = (ROOT / ".github/workflows/release.yml").read_text()
    for reusable in ("release-python.yml", "release-npm.yml", "release-docker.yml", "release-helm.yml"):
        if f"uses: ./.github/workflows/{reusable}" not in top:
            errors.append(f"top-level release does not call {reusable} directly")
    if "needs: [release-please, docker]" not in top:
        errors.append("Helm release must wait for Docker verification")
    for path in sorted((ROOT / ".github/workflows").glob("release*.yml")):
        text = path.read_text()
        if "workflow_dispatch" in text:
            errors.append(f"{path.name} must not expose a publish-capable manual dispatch")
        for token_name in ("PYPI_TOKEN", "NPM_TOKEN", "NODE_AUTH_TOKEN"):
            if token_name in text:
                errors.append(f"{path.name} references forbidden publication token {token_name}")
    for path, environment in (
        (".github/workflows/release-python.yml", "pypi"),
        (".github/workflows/release-npm.yml", "npm"),
    ):
        text = (ROOT / path).read_text()
        if "inputs.publish == true" not in text or "PUBLIC_RELEASE_PUBLISHING_ENABLED == 'true'" not in text:
            errors.append(f"{path} is missing the dual publication latch")
        if f"environment: {environment}" not in text or "id-token: write" not in text:
            errors.append(f"{path} is missing its protected OIDC environment")
    return errors


def tag_errors() -> list[str]:
    result = subprocess.run(
        ["git", "rev-parse", "v0.7.2^{commit}"],  # noqa: S607
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return ["public v0.7.2 tag is missing"]
    ancestor = subprocess.run(  # noqa: S603
        ["git", "merge-base", "--is-ancestor", result.stdout.strip(), "HEAD"],  # noqa: S607
        cwd=ROOT,
        check=False,
    )
    return [] if ancestor.returncode == 0 else ["public v0.7.2 is not an ancestor of HEAD"]


def validate() -> list[str]:
    errors = [*release_config_errors(), *release_workflow_errors(), *workflow_pin_errors(), *tag_errors()]
    if python_matrices() != (PYTHON_DISTRIBUTIONS, PYTHON_DISTRIBUTIONS):
        errors.append("Python build/publish matrices differ from the exact 11-package contract")
    if npm_matrix() != NPM_PACKAGES:
        errors.append("npm publish matrix differs from the exact five-package contract")
    if not (ROOT / "pnpm-lock.yaml").is_file() or (ROOT / "packages/sie_ts_sdk/pnpm-lock.yaml").exists():
        errors.append("root pnpm lock must be the only TypeScript lock authority")
    return errors


def main() -> int:
    errors = validate()
    if errors:
        print("\n".join(f"ERROR: {error}" for error in errors))
        return 1
    print("Public release contract passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
