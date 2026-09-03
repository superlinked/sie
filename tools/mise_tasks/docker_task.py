#!/usr/bin/env python3
"""Public-only Docker build, release-matrix, verification, and alias logic."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from tools.ci.release_artifact import create_manifest, validate_manifest
from tools.ci.release_guard import api, stable_version

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX = ROOT / ".github/release-matrix.json"
DEFAULT_DOCKER_PLATFORM = "linux/amd64"
SOURCE_REVISION = re.compile(r"[0-9a-f]{40}")
VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?")
SERVER_RUST_SUFFIX = "cuda12-sm89"
SINGLETON_DOCKERFILES = {
    "sie-gateway": ROOT / "packages/sie_gateway/Dockerfile",
    "sie-config": ROOT / "packages/sie_config/Dockerfile",
    "sie-mcp": ROOT / "packages/sie_mcp/Dockerfile",
    "sie-server-sidecar": ROOT / "packages/sie_server_sidecar/Dockerfile",
    "sie-server-rust": ROOT / "packages/sie_server_rust/Dockerfile.candle",
    "sie-server-rust-cpu": ROOT / "packages/sie_server_rust/Dockerfile",
}
RELEASE_SINGLETONS = (
    "sie-gateway",
    "sie-config",
    "sie-mcp",
    "sie-server-sidecar",
    "sie-server-rust",
)


@dataclass(frozen=True, order=True)
class ServerTarget:
    platform: str
    bundle: str
    target: str | None = None

    def as_json(self) -> dict[str, str]:
        result = {"platform": self.platform, "bundle": self.bundle}
        if self.target:
            result["target"] = self.target
        return result


def normalize_registry(registry: str) -> str:
    value = registry.strip().rstrip("/")
    if not value or "://" in value:
        raise ValueError("registry must be a non-empty Docker registry/repository prefix")
    return f"{value}/"


def validate_version(version: str) -> str:
    if VERSION.fullmatch(version) is None:
        raise ValueError("version must use an unprefixed semantic version")
    return version


def validate_source_revision(source_revision: str) -> str:
    if SOURCE_REVISION.fullmatch(source_revision) is None:
        raise ValueError("source revision must be a lowercase full 40-character Git SHA")
    return source_revision


def validate_release_versions(version: str) -> None:
    for package in ("sie_server", "sie_sdk", "sie_config", "sie_mcp"):
        path = ROOT / "packages" / package / "pyproject.toml"
        if tomllib.loads(path.read_text())["project"]["version"] != version:
            raise ValueError(f"release version mismatch: {path.relative_to(ROOT)}")
    for package in ("sie_gateway", "sie_server_sidecar", "sie_audio_prep"):
        path = ROOT / "packages" / package / "Cargo.toml"
        if tomllib.loads(path.read_text())["package"]["version"] != version:
            raise ValueError(f"release version mismatch: {path.relative_to(ROOT)}")


def bundle_platform(bundle: str) -> str:
    if re.fullmatch(r"[a-z0-9][a-z0-9-]*", bundle) is None:
        raise ValueError("invalid release bundle name")
    path = ROOT / "packages/sie_server/bundles" / f"{bundle}.yaml"
    if not path.is_file():
        raise ValueError(f"release bundle does not exist: {bundle}")
    data = yaml.safe_load(path.read_text()) or {}
    for adapter in data.get("adapters", []):
        if not isinstance(adapter, str) or not adapter.startswith("sie_server.adapters."):
            raise ValueError(f"bundle {bundle} contains an invalid adapter path")
        module = ROOT / "packages/sie_server/src" / adapter.replace(".", "/")
        if not module.with_suffix(".py").is_file() and not (module / "__init__.py").is_file():
            raise ValueError(f"release bundle {bundle} adapter source is missing: {adapter}")
    declared = data.get("platform", "cuda12")
    if not isinstance(declared, str) or declared not in {"cuda12", "cuda13"}:
        raise ValueError(f"bundle {bundle} declares unsupported platform {declared!r}")
    return declared


def validate_target(target: ServerTarget) -> None:
    declared = bundle_platform(target.bundle)
    if target.platform == "cpu":
        if declared != "cuda12":
            raise ValueError(f"specialized {declared} bundle {target.bundle} has no CPU image")
    elif target.platform != declared:
        raise ValueError(
            f"release target {target.platform}/{target.bundle} disagrees with declared platform {declared}"
        )


def load_release_matrix(path: Path = DEFAULT_MATRIX) -> tuple[ServerTarget, ...]:
    data: dict[str, Any] = json.loads(path.read_text())
    platforms = data.get("platforms")
    bundles = data.get("bundles")
    includes = data.get("include", [])
    if not isinstance(platforms, list) or not isinstance(bundles, list) or not isinstance(includes, list):
        raise ValueError("release matrix must define platforms, bundles, and include lists")
    targets = [ServerTarget(str(platform), str(bundle)) for platform in platforms for bundle in bundles]
    for item in includes:
        if not isinstance(item, dict) or "platform" not in item or "bundle" not in item:
            raise ValueError("release matrix include entries require platform and bundle")
        target_name = item.get("target")
        targets.append(
            ServerTarget(
                str(item["platform"]),
                str(item["bundle"]),
                str(target_name) if target_name is not None else None,
            )
        )
    identities = [(target.platform, target.bundle) for target in targets]
    if len(identities) != len(set(identities)):
        raise ValueError("release matrix contains duplicate platform/bundle targets")
    for target in targets:
        validate_target(target)
    return tuple(targets)


def server_image(registry: str, version: str, target: ServerTarget) -> str:
    return f"{normalize_registry(registry)}sie-server:v{validate_version(version)}-{target.platform}-{target.bundle}"


def singleton_image(registry: str, version: str, service: str) -> str:
    if service not in SINGLETON_DOCKERFILES:
        raise ValueError(f"unknown public image service: {service}")
    suffix = f"-{SERVER_RUST_SUFFIX}" if service == "sie-server-rust" else ""
    image_name = "sie-server-rust" if service == "sie-server-rust-cpu" else service
    if service == "sie-server-rust-cpu":
        suffix = "-cpu"
    return f"{normalize_registry(registry)}{image_name}:v{validate_version(version)}{suffix}"


def expected_versioned_images(
    registry: str,
    version: str,
    targets: tuple[ServerTarget, ...],
) -> tuple[str, ...]:
    images = [server_image(registry, version, target) for target in targets]
    images.extend(singleton_image(registry, version, service) for service in RELEASE_SINGLETONS)
    if len(images) != len(set(images)):
        raise ValueError("release image set contains duplicate tags")
    return tuple(images)


def alias_plan(
    registry: str,
    version: str,
    targets: tuple[ServerTarget, ...],
) -> tuple[tuple[str, str], ...]:
    plan = []
    for target in targets:
        source = server_image(registry, version, target)
        alias = f"{normalize_registry(registry)}sie-server:latest-{target.platform}-{target.bundle}"
        plan.append((source, alias))
    for service in RELEASE_SINGLETONS:
        source = singleton_image(registry, version, service)
        suffix = f"-{SERVER_RUST_SUFFIX}" if service == "sie-server-rust" else ""
        alias = f"{normalize_registry(registry)}{service}:latest{suffix}"
        plan.append((source, alias))
    return tuple(plan)


def build_server_command(
    *,
    registry: str,
    version: str,
    target: ServerTarget,
    source_revision: str,
) -> list[str]:
    validate_target(target)
    revision = validate_source_revision(source_revision)
    dockerfile = ROOT / "packages/sie_server" / f"Dockerfile.{target.platform}"
    if not dockerfile.is_file():
        raise ValueError(f"server Dockerfile does not exist: {dockerfile.relative_to(ROOT)}")
    command = [
        "docker",
        "buildx",
        "build",
        "--platform",
        DEFAULT_DOCKER_PLATFORM,
        "--file",
        str(dockerfile.relative_to(ROOT)),
        "--build-arg",
        f"BUNDLE={target.bundle}",
        "--build-arg",
        f"SIE_SRC_REV={revision}",
        "--label",
        f"org.opencontainers.image.revision={revision}",
        "--label",
        "org.opencontainers.image.source=https://github.com/superlinked/sie",
        "--tag",
        server_image(registry, version, target),
        "--load",
        ".",
    ]
    return command


def build_service_command(
    *,
    registry: str,
    version: str,
    service: str,
    source_revision: str,
) -> list[str]:
    revision = validate_source_revision(source_revision)
    dockerfile = SINGLETON_DOCKERFILES.get(service)
    if dockerfile is None or not dockerfile.is_file():
        raise ValueError(f"Dockerfile is unavailable for {service}")
    command = [
        "docker",
        "buildx",
        "build",
        "--platform",
        DEFAULT_DOCKER_PLATFORM,
        "--file",
        str(dockerfile.relative_to(ROOT)),
    ]
    if service == "sie-server-rust":
        command.extend(["--build-arg", "CUDA_COMPUTE_CAP=89"])
    command.extend(
        [
            "--label",
            f"org.opencontainers.image.revision={revision}",
            "--label",
            "org.opencontainers.image.source=https://github.com/superlinked/sie",
        ]
    )
    command.extend(
        [
            "--tag",
            singleton_image(registry, version, service),
            "--load",
            ".",
        ]
    )
    return command


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)  # noqa: S603


def capture(command: list[str]) -> str:
    return subprocess.check_output(command, cwd=ROOT, text=True).strip()  # noqa: S603


def inspect_loaded(image: str, source_revision: str) -> dict[str, str]:
    records = json.loads(capture(["docker", "image", "inspect", image]))
    if len(records) != 1:
        raise ValueError("expected exactly one loaded image")
    record = records[0]
    labels = record.get("Config", {}).get("Labels", {})
    if labels.get("org.opencontainers.image.revision") != validate_source_revision(source_revision):
        raise ValueError("loaded image source revision mismatch")
    if labels.get("org.opencontainers.image.source") != "https://github.com/superlinked/sie":
        raise ValueError("loaded image source repository mismatch")
    if record.get("Os") != "linux" or record.get("Architecture") != "amd64":
        raise ValueError("release image must be linux/amd64")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", record.get("Id", "")) is None:
        raise ValueError("loaded image has no valid configuration digest")
    return {"image": image, "image_id": record["Id"], "os": "linux", "architecture": "amd64"}


def smoke_image(image: str, *, bundle: str | None = None) -> None:
    command = ["docker", "run", "--rm", "--pull", "never", "--network", "none"]
    if bundle is not None:
        imports = "import sie_server, sie_sdk, sie_audio_prep, torch, transformers; "
        if bundle == "ctranslate2":
            imports += "import ctranslate2; "
        elif bundle in {"sglang", "sglang-cu130"}:
            imports += "import sglang; "
        elif bundle == "tensorrt-llm":
            imports += "import tensorrt_llm; "
        if bundle in {"sglang-cu130", "tensorrt-llm"}:
            imports += "assert torch.version.cuda.startswith('13.'); assert transformers.__version__.startswith('5.'); "
        imports += "print('release image imports passed')"
        command.extend(["--entrypoint", "python", image, "-c", imports])
    else:
        command.extend([image, "--help"])
    run(command)


def export_image(image: str, directory: Path, *, version: str, source_revision: str, run_id: str) -> dict[str, Any]:
    metadata = inspect_loaded(image, source_revision)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("refusing to overwrite an existing image archive")
    run(["docker", "image", "save", "--output", str(directory / "image.tar"), image])
    return create_manifest(
        directory,
        kind="docker",
        version=version,
        tag_name=f"v{version}",
        source_revision=source_revision,
        run_id=run_id,
        metadata=metadata,
    )


def load_image_archive(
    image: str, directory: Path, *, version: str, source_revision: str, run_id: str
) -> dict[str, Any]:
    manifest = validate_manifest(
        directory,
        kind="docker",
        version=version,
        tag_name=f"v{version}",
        source_revision=source_revision,
        run_id=run_id,
    )
    if [item["name"] for item in manifest["files"]] != ["image.tar"]:
        raise ValueError("image archive must contain exactly image.tar")
    if manifest["metadata"].get("image") != image:
        raise ValueError("image archive reference mismatch")
    run(["docker", "image", "load", "--input", str(directory / "image.tar")])
    if inspect_loaded(image, source_revision) != manifest["metadata"]:
        raise ValueError("loaded image configuration digest mismatch")
    return manifest


def remote_image_id(image: str, *, allow_missing: bool = False) -> str | None:
    result = subprocess.run(  # noqa: S603
        ["docker", "buildx", "imagetools", "inspect", "--raw", image],  # noqa: S607
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        if allow_missing and any(
            marker in result.stderr.lower() for marker in ("manifest unknown", "not found", "name unknown")
        ):
            return None
        raise RuntimeError(f"cannot inspect remote image {image}: {result.stderr.strip()}")
    manifest = json.loads(result.stdout)
    if "manifests" in manifest:
        descriptors = [
            item for item in manifest["manifests"] if item.get("platform") == {"architecture": "amd64", "os": "linux"}
        ]
        if len(descriptors) != 1:
            raise ValueError("remote index must contain exactly one linux/amd64 image")
        repository = image.split("@", 1)[0].rsplit(":", 1)[0]
        return remote_image_id(f"{repository}@{descriptors[0]['digest']}")
    digest = manifest.get("config", {}).get("digest", "")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
        raise ValueError("remote image configuration digest is missing")
    return digest


def publish_archive(image: str, directory: Path, *, version: str, source_revision: str, run_id: str) -> None:
    stable_version(version)
    if not image.startswith("ghcr.io/superlinked/") or f":v{version}" not in image:
        raise ValueError("publication requires a versioned public image reference")
    manifest = load_image_archive(image, directory, version=version, source_revision=source_revision, run_id=run_id)
    expected = manifest["metadata"]["image_id"]
    existing = remote_image_id(image, allow_missing=True)
    if existing is not None and existing != expected:
        raise ValueError("refusing to overwrite a different versioned image")
    if existing is None:
        run(["docker", "push", image])
    if remote_image_id(image) != expected:
        raise ValueError("remote image does not match the tested image configuration digest")


def verify_release(
    registry: str,
    version: str,
    targets: tuple[ServerTarget, ...],
    *,
    evidence_dir: Path,
    source_revision: str,
    run_id: str,
) -> None:
    images = set(expected_versioned_images(registry, version, targets))
    evidence = {}
    for path in evidence_dir.rglob("*.json"):
        manifest = json.loads(path.read_text())
        expected_identity = {
            "schema": 1,
            "repository": "superlinked/sie",
            "kind": "docker",
            "version": version,
            "tag_name": f"v{version}",
            "source_revision": validate_source_revision(source_revision),
            "run_id": str(run_id),
        }
        if any(manifest.get(key) != value for key, value in expected_identity.items()):
            raise ValueError("image evidence source/run identity mismatch")
        metadata = manifest["metadata"]
        image = metadata["image"]
        if metadata.get("architecture") != "amd64" or metadata.get("os") != "linux" or image in evidence:
            raise ValueError("duplicate or wrong-platform image evidence")
        evidence[image] = metadata["image_id"]
    if set(evidence) != images:
        raise ValueError("image evidence does not cover the exact complete release set")
    for image, image_id in evidence.items():
        if remote_image_id(image) != image_id:
            raise ValueError(f"remote image differs from tested source-bound image: {image}")


def published_tag_revision(tag: str) -> str:
    reference = api(f"git/ref/tags/{tag}")
    if not isinstance(reference, dict) or reference.get("ref") != f"refs/tags/{tag}":
        raise ValueError("published release tag reference mismatch")
    obj = reference.get("object", {})
    for _ in range(5):
        if not isinstance(obj, dict) or not isinstance(obj.get("sha"), str):
            raise ValueError("published release tag object is malformed")
        revision = validate_source_revision(obj.get("sha", ""))
        if obj.get("type") == "commit":
            return revision
        if obj.get("type") != "tag":
            raise ValueError("published release tag does not resolve to a commit")
        annotated = api(f"git/tags/{revision}")
        if not isinstance(annotated, dict) or annotated.get("sha") != revision or annotated.get("tag") != tag:
            raise ValueError("published annotated tag identity mismatch")
        obj = annotated.get("object", {})
    raise ValueError("published release tag chain does not resolve to a commit")


def alias_release_is_current(version: str, source_revision: str) -> bool:
    requested = stable_version(version)
    revision = validate_source_revision(source_revision)
    pages = json.loads(capture(["gh", "api", "--paginate", "--slurp", "repos/superlinked/sie/releases?per_page=100"]))
    if not isinstance(pages, list) or any(not isinstance(page, list) for page in pages):
        raise ValueError("published release listing is malformed")
    releases = {}
    for release in (release for page in pages for release in page):
        if (
            not isinstance(release, dict)
            or not isinstance(release.get("draft"), bool)
            or not isinstance(release.get("prerelease"), bool)
        ):
            raise ValueError("published release state is malformed")
        if release["draft"] or release["prerelease"]:
            continue
        tag = release.get("tag_name")
        if not isinstance(tag, str) or not tag.startswith("v") or not release.get("published_at"):
            raise ValueError("published stable release identity is malformed")
        try:
            datetime.strptime(release["published_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        except (TypeError, ValueError) as error:
            raise ValueError("published stable release timestamp is malformed") from error
        released = stable_version(tag[1:], new=False)
        if released in releases:
            raise ValueError("published stable release identity is duplicated")
        releases[released] = tag
    if requested not in releases:
        raise ValueError("requested release is absent from published stable releases")
    if published_tag_revision(releases[requested]) != revision:
        raise ValueError("requested published release tag differs from its source revision")
    latest = max(releases)
    if latest == requested:
        return True
    latest_revision = published_tag_revision(releases[latest])
    comparison = api(f"compare/{revision}...{latest_revision}")
    if not isinstance(comparison, dict) or comparison.get("status") != "ahead":
        raise ValueError("newer published release does not descend from the requested release")
    print(f"Leaving floating aliases unchanged: newer stable release {releases[latest]} is published")
    return False


def move_aliases(registry: str, version: str, targets: tuple[ServerTarget, ...], **kwargs: Any) -> None:
    verify_release(registry, version, targets, **kwargs)
    if not alias_release_is_current(version, kwargs["source_revision"]):
        return
    for source, alias in alias_plan(registry, version, targets):
        run(["docker", "buildx", "imagetools", "create", "--tag", alias, source])


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    subparsers = result.add_subparsers(dest="command", required=True)
    matrix = subparsers.add_parser("matrix")
    matrix.add_argument("--file", type=Path, default=DEFAULT_MATRIX)
    matrix.add_argument("--version")

    for name in ("expected", "verify", "alias"):
        command = subparsers.add_parser(name)
        command.add_argument("--registry", required=True)
        command.add_argument("--version", required=True)
        command.add_argument("--matrix-file", type=Path, default=DEFAULT_MATRIX)
        if name != "expected":
            command.add_argument("--evidence-dir", type=Path, required=True)
            command.add_argument("--source-revision", required=True)
            command.add_argument("--run-id", required=True)

    server = subparsers.add_parser("build-server")
    server.add_argument("--registry", required=True)
    server.add_argument("--version", required=True)
    server.add_argument("--platform", required=True)
    server.add_argument("--bundle", required=True)
    server.add_argument("--source-revision", required=True)

    service = subparsers.add_parser("build-service")
    service.add_argument("--registry", required=True)
    service.add_argument("--version", required=True)
    service.add_argument("--service", choices=sorted(SINGLETON_DOCKERFILES), required=True)
    service.add_argument("--source-revision", required=True)
    for command in (server, service):
        command.add_argument("--archive-dir", type=Path)
        command.add_argument("--evidence-dir", type=Path)
        command.add_argument("--run-id")

    for name in ("load", "publish"):
        command = subparsers.add_parser(name)
        command.add_argument("--image", required=True)
        command.add_argument("--archive-dir", type=Path, required=True)
        command.add_argument("--version", required=True)
        command.add_argument("--source-revision", required=True)
        command.add_argument("--run-id", required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command in {"build-server", "build-service"} and args.archive_dir and not args.run_id:
            raise ValueError("archiving requires the original Actions run ID")
        if args.command == "matrix":
            print(json.dumps({"include": [item.as_json() for item in load_release_matrix(args.file)]}))
            if args.version:
                validate_release_versions(args.version)
        elif args.command == "expected":
            targets = load_release_matrix(args.matrix_file)
            print("\n".join(expected_versioned_images(args.registry, args.version, targets)))
        elif args.command == "build-server":
            run(
                build_server_command(
                    registry=args.registry,
                    version=args.version,
                    target=ServerTarget(args.platform, args.bundle),
                    source_revision=args.source_revision,
                )
            )
        elif args.command == "build-service":
            run(
                build_service_command(
                    registry=args.registry,
                    version=args.version,
                    service=args.service,
                    source_revision=args.source_revision,
                )
            )
        elif args.command in {"load", "publish"}:
            operation = load_image_archive if args.command == "load" else publish_archive
            operation(
                args.image,
                args.archive_dir,
                version=args.version,
                source_revision=args.source_revision,
                run_id=args.run_id,
            )
        elif args.command == "verify":
            verify_release(
                args.registry,
                args.version,
                load_release_matrix(args.matrix_file),
                evidence_dir=args.evidence_dir,
                source_revision=args.source_revision,
                run_id=args.run_id,
            )
        elif args.command == "alias":
            move_aliases(
                args.registry,
                args.version,
                load_release_matrix(args.matrix_file),
                evidence_dir=args.evidence_dir,
                source_revision=args.source_revision,
                run_id=args.run_id,
            )
        if args.command in {"build-server", "build-service"} and args.archive_dir:
            image = (
                server_image(args.registry, args.version, ServerTarget(args.platform, args.bundle))
                if args.command == "build-server"
                else singleton_image(args.registry, args.version, args.service)
            )
            smoke_image(image, bundle=args.bundle if args.command == "build-server" else None)
            export_image(
                image, args.archive_dir, version=args.version, source_revision=args.source_revision, run_id=args.run_id
            )
            if args.evidence_dir:
                args.evidence_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(args.archive_dir / "provenance.json", args.evidence_dir / "provenance.json")
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"docker task failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
