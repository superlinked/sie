"""Install docker buildx into ~/.docker/cli-plugins.

Builder creation is an explicit opt-in because it pulls a BuildKit image and
changes Docker's active global builder.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from urllib.request import urlopen

BUILDX_VERSION = os.environ.get("BUILDX_VERSION", "0.29.1")
DOWNLOAD_TIMEOUT_SECONDS = 30
MAX_BINARY_BYTES = 128 * 1024 * 1024
MAX_CHECKSUM_BYTES = 1024 * 1024
CHECKSUM_FIELD_COUNT = 2
SHA256_HEX_LENGTH = 64


def detect_arch() -> str:
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        return "amd64"
    if machine in {"aarch64", "arm64"}:
        return "arm64"
    raise RuntimeError(f"Unsupported architecture for buildx: {machine}")


def detect_os() -> str:
    system = platform.system().lower()
    if system == "linux":
        return "linux"
    if system == "darwin":
        return "darwin"
    raise RuntimeError(f"Unsupported operating system for buildx: {system}")


def buildx_asset_name() -> str:
    return f"buildx-v{BUILDX_VERSION}.{detect_os()}-{detect_arch()}"


def docker_available() -> bool:
    return shutil.which("docker") is not None


def buildx_installed() -> bool:
    result = subprocess.run(
        ["docker", "buildx", "version"],  # noqa: S607 — intentional partial path
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def builder_exists(name: str) -> bool:
    """Check if a buildx builder with the given name exists."""
    result = subprocess.run(  # noqa: S603 — intentional subprocess call
        ["docker", "buildx", "inspect", name],  # noqa: S607 — intentional partial path
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def ensure_docker_container_builder() -> None:
    """Create a docker-container builder if one doesn't exist.

    The docker-container driver is required for cache export/import,
    which significantly speeds up repeated builds.
    """
    builder_name = "sie-builder"

    if builder_exists(builder_name):
        print(f"Builder '{builder_name}' already exists.")
        # Ensure it's the active builder
        subprocess.run(["docker", "buildx", "use", builder_name], check=False)  # noqa: S607, S603 — intentional partial path
        return

    print(f"Creating docker-container builder '{builder_name}'...")
    result = subprocess.run(  # noqa: S603 — intentional subprocess call
        [  # noqa: S607 — intentional partial path
            "docker",
            "buildx",
            "create",
            "--name",
            builder_name,
            "--driver",
            "docker-container",
            "--use",
        ],
        check=False,
    )
    if result.returncode != 0:
        print("Warning: Failed to create docker-container builder")
        print("Cache export will not work, but builds will still succeed")
        return

    # Bootstrap the builder (pulls the buildkit image)
    print("Bootstrapping builder...")
    subprocess.run(["docker", "buildx", "inspect", "--bootstrap"], check=False)  # noqa: S607 — intentional partial path
    print(f"Builder '{builder_name}' created and activated.")


def download_bytes(url: str, *, max_bytes: int) -> bytes:
    """Download at most ``max_bytes`` from one trusted release URL."""
    with urlopen(  # noqa: S310 — caller supplies a pinned GitHub release URL.
        url,
        timeout=DOWNLOAD_TIMEOUT_SECONDS,
    ) as response:
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise RuntimeError(f"Download exceeds the {max_bytes}-byte safety limit: {url}")
    return data


def checksum_for_asset(checksums: str, asset_name: str) -> str:
    """Return the SHA256 for an exact checksum-manifest filename."""
    for line in checksums.splitlines():
        parts = line.split()
        if len(parts) != CHECKSUM_FIELD_COUNT:
            continue
        digest, filename = parts
        if filename.lstrip("*") == asset_name and len(digest) == SHA256_HEX_LENGTH:
            return digest
    raise RuntimeError(f"Could not find checksum for {asset_name} in checksums.txt")


def install_verified_binary(plugin_path: Path, data: bytes) -> None:
    """Atomically replace the plugin without following an existing symlink."""
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=".docker-buildx.", dir=plugin_path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as temporary_file:
            temporary_file.write(data)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.chmod(temporary_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        temporary_path.replace(plugin_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configure-builder",
        action="store_true",
        help="create/activate the global sie-builder and bootstrap BuildKit",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not docker_available():
        print("Docker is not installed; skipping optional buildx setup.")
        return 0

    if buildx_installed():
        print("buildx already installed.")
        if args.configure_builder:
            ensure_docker_container_builder()
        return 0

    plugin_dir = Path.home() / ".docker" / "cli-plugins"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    plugin_path = plugin_dir / "docker-buildx"

    binary_name = buildx_asset_name()
    url = f"https://github.com/docker/buildx/releases/download/v{BUILDX_VERSION}/{binary_name}"
    print(f"Downloading buildx v{BUILDX_VERSION} ({binary_name})...")
    data = download_bytes(url, max_bytes=MAX_BINARY_BYTES)

    # Verify SHA256 checksum (buildx uses a single checksums.txt file)
    checksums_url = f"https://github.com/docker/buildx/releases/download/v{BUILDX_VERSION}/checksums.txt"
    print("Verifying SHA256 checksum...")
    checksums = download_bytes(checksums_url, max_bytes=MAX_CHECKSUM_BYTES).decode()
    expected_hash = checksum_for_asset(checksums, binary_name)
    actual_hash = hashlib.sha256(data).hexdigest()
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"SHA256 checksum mismatch for buildx download.\n  Expected: {expected_hash}\n  Actual:   {actual_hash}"
        )

    install_verified_binary(plugin_path, data)
    print(f"Installed buildx to {plugin_path}")

    if args.configure_builder:
        ensure_docker_container_builder()
    return 0


if __name__ == "__main__":
    sys.exit(main())
