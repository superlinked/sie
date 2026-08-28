#!/usr/bin/env python3
"""Validate the public version, artifact, and fail-closed publication contract."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import tomllib
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
PUBLIC_IMAGE_NAMES = {
    "sie-server",
    "sie-gateway",
    "sie-config",
    "sie-mcp",
    "sie-server-sidecar",
    "sie-server-rust",
}
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

CANDLE_PATHS = (
    "packages/sie_server_rust/Cargo.lock",
    "packages/sie_server_rust/Cargo.toml",
    "packages/sie_server_rust/Dockerfile",
    "packages/sie_server_rust/Dockerfile.candle",
    "packages/sie_server_rust/src/candle_backend.rs",
    "packages/sie_server_rust/src/candle_bert_flash.rs",
    "packages/sie_server_rust/src/candle_embedding.rs",
    "packages/sie_server_rust/src/candle_gte_rope.rs",
    "packages/sie_server_rust/src/candle_layers.rs",
    "packages/sie_server_rust/src/candle_modernbert.rs",
    "packages/sie_server_rust/src/candle_residency.rs",
    "packages/sie_server_rust/src/candle_rope.rs",
    "packages/sie_server_rust/src/candle_splade.rs",
    "packages/sie_server_rust/src/candle_xlm_roberta.rs",
    "packages/sie_server_rust/src/ipc.rs",
    "packages/sie_server_rust/src/ipc_types.rs",
    "packages/sie_server_rust/src/lib.rs",
    "packages/sie_server_rust/src/main.rs",
    "packages/sie_server_rust/src/native_backend.rs",
    "packages/sie_server_rust/src/observability/metrics.rs",
    "packages/sie_server_rust/src/observability/mod.rs",
    "packages/sie_server_rust/src/observability/propagation.rs",
    "packages/sie_server_rust/src/observability/resource.rs",
    "packages/sie_server_rust/src/observability/tracing.rs",
    "packages/sie_server_rust/src/observability/transport.rs",
    "packages/sie_server_rust/src/text_prep.rs",
    "packages/sie_server_rust/vendor/candle-cublaslt/Cargo.toml",
    "packages/sie_server_rust/vendor/candle-cublaslt/LICENSE-APACHE",
    "packages/sie_server_rust/vendor/candle-cublaslt/LICENSE-MIT",
    "packages/sie_server_rust/vendor/candle-cublaslt/README.md",
    "packages/sie_server_rust/vendor/candle-cublaslt/src/lib.rs",
    "packages/sie_server_rust/vendor/candle-gated-activation/Cargo.toml",
    "packages/sie_server_rust/vendor/candle-gated-activation/build.rs",
    "packages/sie_server_rust/vendor/candle-gated-activation/kernels/gated_activation.cu",
    "packages/sie_server_rust/vendor/candle-gated-activation/kernels/gelu_erf_gate.cu",
    "packages/sie_server_rust/vendor/candle-gated-activation/src/ffi.rs",
    "packages/sie_server_rust/vendor/candle-gated-activation/src/lib.rs",
    "packages/sie_server_rust/vendor/candle-layer-norm/Cargo.toml",
    "packages/sie_server_rust/vendor/candle-layer-norm/LICENSE",
    "packages/sie_server_rust/vendor/candle-layer-norm/LICENSE-APACHE",
    "packages/sie_server_rust/vendor/candle-layer-norm/LICENSE-MIT",
    "packages/sie_server_rust/vendor/candle-layer-norm/README.md",
    "packages/sie_server_rust/vendor/candle-layer-norm/build.rs",
    "packages/sie_server_rust/vendor/candle-layer-norm/kernels/ln.h",
    "packages/sie_server_rust/vendor/candle-layer-norm/kernels/ln_api.cu",
    "packages/sie_server_rust/vendor/candle-layer-norm/kernels/ln_fwd_kernels.cuh",
    "packages/sie_server_rust/vendor/candle-layer-norm/kernels/ln_kernel_traits.h",
    "packages/sie_server_rust/vendor/candle-layer-norm/kernels/ln_utils.cuh",
    "packages/sie_server_rust/vendor/candle-layer-norm/kernels/static_switch.h",
    "packages/sie_server_rust/vendor/candle-layer-norm/src/ffi.rs",
    "packages/sie_server_rust/vendor/candle-layer-norm/src/lib.rs",
    "packages/sie_server_rust/vendor/candle-rotary/Cargo.toml",
    "packages/sie_server_rust/vendor/candle-rotary/LICENSE-APACHE",
    "packages/sie_server_rust/vendor/candle-rotary/LICENSE-MIT",
    "packages/sie_server_rust/vendor/candle-rotary/README.md",
    "packages/sie_server_rust/vendor/candle-rotary/build.rs",
    "packages/sie_server_rust/vendor/candle-rotary/kernels/cuda_compat.h",
    "packages/sie_server_rust/vendor/candle-rotary/kernels/rotary.cu",
    "packages/sie_server_rust/vendor/candle-rotary/src/ffi.rs",
    "packages/sie_server_rust/vendor/candle-rotary/src/lib.rs",
    "packages/sie_server_rust/vendor/candle-rotary/tests/rotary_tests.rs",
    "packages/sie_server_rust/vendor/candle-splade-pool/Cargo.toml",
    "packages/sie_server_rust/vendor/candle-splade-pool/build.rs",
    "packages/sie_server_rust/vendor/candle-splade-pool/kernels/splade_pool.cu",
    "packages/sie_server_rust/vendor/candle-splade-pool/src/ffi.rs",
    "packages/sie_server_rust/vendor/candle-splade-pool/src/lib.rs",
)


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


def candle_source_errors() -> list[str]:
    errors: list[str] = []
    root = ROOT / "packages/sie_server_rust"
    listed = subprocess.run(
        [
            "/usr/bin/git",
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "packages/sie_server_rust",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    actual = {item.decode() for item in listed.stdout.split(b"\0") if item}
    expected = set(CANDLE_PATHS)
    if actual != expected:
        errors.append(
            "Candle source closure differs from the reviewed 66-path allowlist: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )
        return errors
    for relative in sorted(expected):
        path = ROOT / relative
        if path.is_symlink():
            errors.append(f"Candle source closure must not contain symlinks: {relative}")
        try:
            path.read_text()
        except UnicodeDecodeError:
            errors.append(f"Candle source closure contains non-text content: {relative}")

    required_licenses = {
        "packages/sie_server_rust/vendor/candle-cublaslt/LICENSE-APACHE",
        "packages/sie_server_rust/vendor/candle-cublaslt/LICENSE-MIT",
        "packages/sie_server_rust/vendor/candle-layer-norm/LICENSE",
        "packages/sie_server_rust/vendor/candle-layer-norm/LICENSE-APACHE",
        "packages/sie_server_rust/vendor/candle-layer-norm/LICENSE-MIT",
        "packages/sie_server_rust/vendor/candle-rotary/LICENSE-APACHE",
        "packages/sie_server_rust/vendor/candle-rotary/LICENSE-MIT",
    }
    if not required_licenses.issubset(actual):
        errors.append("Candle vendored license set is incomplete")

    def walk_paths(value: Any) -> list[str]:
        found: list[str] = []
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "path" and isinstance(item, str):
                    found.append(item)
                else:
                    found.extend(walk_paths(item))
        elif isinstance(value, list):
            for item in value:
                found.extend(walk_paths(item))
        return found

    for manifest in root.rglob("Cargo.toml"):
        data = tomllib.loads(manifest.read_text())
        for dependency_path in walk_paths(data):
            if not (manifest.parent / dependency_path).resolve().exists():
                errors.append(
                    f"Candle manifest path does not resolve: {manifest.relative_to(ROOT)} -> {dependency_path}"
                )
    return errors


def docker_copy_errors() -> list[str]:
    errors: list[str] = []
    dockerfiles = [
        ROOT / "packages/sie_server/Dockerfile.cpu",
        ROOT / "packages/sie_server/Dockerfile.cuda12",
        ROOT / "packages/sie_server/Dockerfile.cuda13",
        ROOT / "packages/sie_gateway/Dockerfile",
        ROOT / "packages/sie_config/Dockerfile",
        ROOT / "packages/sie_mcp/Dockerfile",
        ROOT / "packages/sie_server_sidecar/Dockerfile",
        ROOT / "packages/sie_server_rust/Dockerfile",
        ROOT / "packages/sie_server_rust/Dockerfile.candle",
    ]
    for dockerfile in dockerfiles:
        logical_text = dockerfile.read_text().replace("\\\n", " ")
        for line_number, line in enumerate(logical_text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped.startswith("COPY "):
                continue
            tokens = shlex.split(stripped)
            if any(token.startswith("--from=") for token in tokens[1:]):
                continue
            arguments = [token for token in tokens[1:] if not token.startswith("--")]
            for source in arguments[:-1]:
                if source.startswith("/") or "$" in source:
                    errors.append(
                        f"{dockerfile.relative_to(ROOT)}:{line_number}: unsupported release COPY source {source}"
                    )
                    continue
                matches = list(ROOT.glob(source))
                if not matches:
                    errors.append(f"{dockerfile.relative_to(ROOT)}:{line_number}: missing release COPY source {source}")
        for line in logical_text.splitlines():
            if (
                "org.opencontainers.image.source=" in line
                and 'org.opencontainers.image.source="https://github.com/superlinked/sie"' not in line
            ):
                errors.append(f"{dockerfile.relative_to(ROOT)} has a non-public OCI source label")
    return errors


def docker_release_errors() -> list[str]:
    errors = [*candle_source_errors(), *docker_copy_errors()]
    matrix = load_json(".github/release-matrix.json")
    pairs = {(platform, bundle) for platform in matrix.get("platforms", []) for bundle in matrix.get("bundles", [])}
    pairs.update((item.get("platform"), item.get("bundle")) for item in matrix.get("include", []))
    expected_pairs = {
        (platform, bundle) for platform in ("cuda12", "cpu") for bundle in ("default", "sglang", "transformers5")
    } | {("cuda13", "sglang-cu130")}
    if pairs != expected_pairs:
        errors.append("Docker release matrix differs from the supported server pairs")

    values = (ROOT / "deploy/helm/sie-cluster/values.yaml").read_text()
    chart_images = set(re.findall(r"repository:\s*ghcr\.io/superlinked/(sie-[a-z-]+)", values))
    if chart_images != PUBLIC_IMAGE_NAMES:
        errors.append(f"chart-advertised SIE repositories differ from release set: {sorted(chart_images)}")

    workflow = (ROOT / ".github/workflows/release-docker.yml").read_text()
    if "inputs.publish == true" not in workflow or "PUBLIC_RELEASE_PUBLISHING_ENABLED == 'true'" not in workflow:
        errors.append("Docker release is missing its dual publication latch")
    if "needs: [matrix, verify]" not in workflow:
        errors.append("Docker latest aliases are not ordered after full-set verification")
    return errors


def helm_release_errors() -> list[str]:
    errors: list[str] = []
    chart = (ROOT / "deploy/helm/sie-cluster/Chart.yaml").read_text()
    version_match = re.search(r"^version:\s*([^\s#]+)", chart, re.MULTILINE)
    app_match = re.search(r"^appVersion:\s*([^\s#]+)", chart, re.MULTILINE)
    if not version_match or not app_match or app_match.group(1) != f"v{version_match.group(1)}":
        errors.append("Helm Chart version and appVersion must share the vX.Y.Z release identity")

    workflow = (ROOT / ".github/workflows/release-helm.yml").read_text()
    required = (
        "ref: ${{ inputs.sha }}",
        "mise run helm -- dependencies",
        "mise run helm -- lint --set payloadStore.enabled=false",
        "mise run helm -- template --set payloadStore.enabled=false",
        "helm package deploy/helm/sie-cluster",
        "needs: build",
        "inputs.publish == true",
        "PUBLIC_RELEASE_PUBLISHING_ENABLED == 'true'",
        "packages: write",
        "helm push",
        "helm show chart",
    )
    missing = [item for item in required if item not in workflow]
    if missing:
        errors.append(f"Helm release workflow is missing contract elements: {missing}")
    if "latest" in workflow or "alias" in workflow:
        errors.append("Helm release must not move a floating chart alias")

    docker_task = (ROOT / "tools/mise_tasks/docker_task.py").read_text()
    if ":v{validate_version(version)}" not in docker_task:
        errors.append("Docker versioned tags must match the chart's v-prefixed appVersion")
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
    errors = [
        *release_config_errors(),
        *release_workflow_errors(),
        *docker_release_errors(),
        *helm_release_errors(),
        *workflow_pin_errors(),
        *tag_errors(),
    ]
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
