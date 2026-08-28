#!/usr/bin/env python3
"""Public-only Docker build, release-matrix, verification, and alias logic."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

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


def bundle_platform(bundle: str) -> str:
    path = ROOT / "packages/sie_server/bundles" / f"{bundle}.yaml"
    if not path.is_file():
        raise ValueError(f"release bundle does not exist: {bundle}")
    data = yaml.safe_load(path.read_text()) or {}
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
    push: bool,
) -> list[str]:
    validate_target(target)
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
        f"SIE_SRC_REV={validate_source_revision(source_revision)}",
        "--tag",
        server_image(registry, version, target),
        "--push" if push else "--load",
        ".",
    ]
    return command


def build_service_command(
    *,
    registry: str,
    version: str,
    service: str,
    source_revision: str,
    push: bool,
) -> list[str]:
    validate_source_revision(source_revision)
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
            "--tag",
            singleton_image(registry, version, service),
            "--push" if push else "--load",
            ".",
        ]
    )
    return command


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)  # noqa: S603


def image_exists(image: str, *, attempts: int = 4) -> bool:
    for attempt in range(attempts):
        result = subprocess.run(  # noqa: S603
            ["docker", "buildx", "imagetools", "inspect", image],  # noqa: S607
            cwd=ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            return True
        if attempt + 1 < attempts:
            time.sleep(2**attempt)
    return False


def verify_release(registry: str, version: str, targets: tuple[ServerTarget, ...]) -> None:
    images = expected_versioned_images(registry, version, targets)
    missing = [image for image in images if not image_exists(image)]
    if missing:
        raise RuntimeError("missing versioned release images:\n" + "\n".join(missing))


def move_aliases(registry: str, version: str, targets: tuple[ServerTarget, ...]) -> None:
    verify_release(registry, version, targets)
    for source, alias in alias_plan(registry, version, targets):
        run(["docker", "buildx", "imagetools", "create", "--tag", alias, source])


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    subparsers = result.add_subparsers(dest="command", required=True)
    matrix = subparsers.add_parser("matrix")
    matrix.add_argument("--file", type=Path, default=DEFAULT_MATRIX)

    for name in ("expected", "verify", "alias"):
        command = subparsers.add_parser(name)
        command.add_argument("--registry", required=True)
        command.add_argument("--version", required=True)
        command.add_argument("--matrix-file", type=Path, default=DEFAULT_MATRIX)

    server = subparsers.add_parser("build-server")
    server.add_argument("--registry", required=True)
    server.add_argument("--version", required=True)
    server.add_argument("--platform", required=True)
    server.add_argument("--bundle", required=True)
    server.add_argument("--source-revision", required=True)
    server.add_argument("--push", action="store_true")

    service = subparsers.add_parser("build-service")
    service.add_argument("--registry", required=True)
    service.add_argument("--version", required=True)
    service.add_argument("--service", choices=sorted(SINGLETON_DOCKERFILES), required=True)
    service.add_argument("--source-revision", required=True)
    service.add_argument("--push", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "matrix":
            print(json.dumps({"include": [item.as_json() for item in load_release_matrix(args.file)]}))
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
                    push=args.push,
                )
            )
        elif args.command == "build-service":
            run(
                build_service_command(
                    registry=args.registry,
                    version=args.version,
                    service=args.service,
                    source_revision=args.source_revision,
                    push=args.push,
                )
            )
        elif args.command == "verify":
            verify_release(args.registry, args.version, load_release_matrix(args.matrix_file))
        elif args.command == "alias":
            move_aliases(args.registry, args.version, load_release_matrix(args.matrix_file))
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"docker task failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
