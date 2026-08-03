# ruff: noqa: INP001
"""Build the pinned native audio-preprocessing wheel for worker images."""

from __future__ import annotations

import fcntl
import hashlib
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

MATURIN_VERSION = "1.13.3"
AUDIO_PREP_VERSION = "0.6.26"  # x-release-please-version
AUDIO_WHEEL_COMPATIBILITY = "manylinux_2_28"
AUDIO_WHEEL_FILENAME = f"sie_audio_prep-{AUDIO_PREP_VERSION}-cp312-abi3-{AUDIO_WHEEL_COMPATIBILITY}_x86_64.whl"
AUDIO_WHEEL_REMOTE = f"/opt/sie/wheels/{AUDIO_WHEEL_FILENAME}"
_BUILD_FINGERPRINT = f"maturin={MATURIN_VERSION};compatibility={AUDIO_WHEEL_COMPATIBILITY};zig=true"


def _source_digest(crate: Path) -> str:
    digest = hashlib.sha256()
    digest.update(_BUILD_FINGERPRINT.encode())
    # The crate is a workspace member (#2339): its dependency pins live in
    # the workspace-root Cargo.lock, one level above packages/<crate>. A
    # missing lock must fail loudly — silently skipping it would drop
    # dependency pins from the wheel cache fingerprint.
    workspace_lock = crate.parents[1] / "Cargo.lock"
    if not workspace_lock.is_file():
        raise RuntimeError(f"workspace Cargo.lock not found at {workspace_lock}")
    paths = [crate / "Cargo.toml", workspace_lock, crate / "pyproject.toml"]
    paths.extend(sorted((crate / "src").rglob("*")))
    for path in paths:
        if not path.is_file():
            continue
        try:
            relative = path.relative_to(crate).as_posix().encode()
        except ValueError:  # workspace-root lock, above the crate dir
            relative = f"../../{path.name}".encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _validate_wheel(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        mode = 0
    if not stat.S_ISREG(mode) or path.suffix != ".whl":
        raise RuntimeError(f"audio prep build did not produce a wheel: {path}")
    if path.name != AUDIO_WHEEL_FILENAME:
        raise RuntimeError(f"unexpected sie-audio-prep wheel tag: {path.name}")
    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
        metadata_paths = [name for name in names if name.endswith(".dist-info/METADATA")]
        extensions = [name for name in names if name.startswith("sie_audio_prep") and name.endswith(".so")]
        if len(metadata_paths) != 1 or not extensions:
            raise RuntimeError(f"invalid sie-audio-prep wheel contents: {path}")
        metadata = wheel.read(metadata_paths[0]).decode("utf-8")
    if "Name: sie-audio-prep\n" not in metadata or f"Version: {AUDIO_PREP_VERSION}\n" not in metadata:
        raise RuntimeError(f"unexpected sie-audio-prep wheel metadata: {path}")


def _ensure_private_cache_dir(path: Path) -> None:
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        pass  # The existing node is validated with non-following lstat below.
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise RuntimeError(f"unsafe audio prep wheel cache directory: {path}")


def build_audio_prep_wheel(project_root: Path, *, required: bool = True) -> Path | None:
    """Build once per source hash and return a validated Linux x86_64 wheel.

    ``required=False`` is for opportunistic consumers (the Modal sandbox
    image): a host that cannot build the wheel -- non-Linux, or uvx/zig not
    on PATH -- gets None plus a stderr note, and the image ships without
    audio support (the audio adapters fail closed at runtime). Deploy paths
    keep the default and fail at build time.
    """
    if sys.platform != "linux" or platform.machine() not in {"x86_64", "AMD64"}:
        if not required:
            print("sie-audio-prep wheel skipped: audio wheels build on Linux x86_64 only", file=sys.stderr)
            return None
        raise RuntimeError("audio worker images must be deployed from Linux x86_64")

    crate = project_root / "packages" / "sie_audio_prep"
    digest = _source_digest(crate)
    cache_root = Path(tempfile.gettempdir()) / f"sie-audio-prep-wheels-{os.getuid()}"
    _ensure_private_cache_dir(cache_root)
    output_dir = cache_root / digest
    _ensure_private_cache_dir(output_dir)
    lock_path = output_dir / ".build.lock"
    lock_fd = os.open(lock_path, os.O_CLOEXEC | os.O_CREAT | os.O_NOFOLLOW | os.O_RDWR, 0o600)
    with os.fdopen(lock_fd, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        wheels = sorted(output_dir.glob("*.whl"))
        if not wheels:
            uvx = shutil.which("uvx")
            zig = shutil.which("zig")
            if uvx is None or zig is None:
                missing = " and ".join(name for name, path in (("uvx", uvx), ("zig", zig)) if path is None)
                if not required:
                    print(f"sie-audio-prep wheel skipped: {missing} not on PATH", file=sys.stderr)
                    return None
                raise RuntimeError(f"{missing} required to build portable sie-audio-prep wheels")
            subprocess.run(  # noqa: S603
                [
                    uvx,
                    "--from",
                    f"maturin=={MATURIN_VERSION}",
                    "maturin",
                    "build",
                    "--release",
                    "--locked",
                    "--zig",
                    "--compatibility",
                    AUDIO_WHEEL_COMPATIBILITY,
                    "--out",
                    str(output_dir),
                ],
                check=True,
                cwd=crate,
            )
            wheels = sorted(output_dir.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"expected one sie-audio-prep wheel in {output_dir}, found {len(wheels)}")
        _validate_wheel(wheels[0])
        return wheels[0]
